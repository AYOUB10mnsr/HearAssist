"""YAMNet inference for mono float32 PCM at 16 kHz, with incremental input."""
import zipfile
from time import perf_counter
import numpy as np
from ai_edge_litert.interpreter import Interpreter

TARGETS = {
    'dog': {'fr': 'Aboiement', 'labels': ['Dog', 'Bark']},
    'crying_baby': {'fr': 'Pleurs de bébé', 'labels': ['Baby cry, infant cry']},
    'door_wood_knock': {'fr': 'Coups à la porte', 'labels': ['Knock']},
    'siren': {'fr': 'Sirène', 'labels': ['Siren', 'Police car (siren)', 'Ambulance (siren)', 'Fire engine, fire truck (siren)', 'Civil defense siren']},
    'car_horn': {'fr': 'Klaxon', 'labels': ['Vehicle horn, car horn, honking']},
    'clock_alarm': {'fr': 'Réveil', 'labels': ['Alarm clock']},
    'glass_breaking': {'fr': 'Bris de verre', 'labels': ['Shatter']},
    'engine': {'fr': 'Moteur', 'labels': ['Engine', 'Light engine (high frequency)', 'Medium engine (mid frequency)', 'Heavy engine (low frequency)', 'Engine starting', 'Engine knocking']},
}


class YamnetDetector:
    """One instance per stream. feed() returns completed analysis windows."""
    sample_rate = 16000

    def __init__(self, model_path, hop_seconds=0.48, threads=2):
        with zipfile.ZipFile(model_path) as archive:
            self.labels = archive.read('yamnet_label_list.txt').decode().splitlines()
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.window_samples = int(np.prod(self.input['shape']))
        self.hop_samples = round(hop_seconds * self.sample_rate)
        if not 0 < self.hop_samples <= self.window_samples:
            raise ValueError('Hop must be positive and no longer than the input window.')
        if self.output['shape'][-1] != len(self.labels):
            raise ValueError('Model output and labels disagree.')
        self.categories = list(TARGETS)
        self.groups = [[self.labels.index(label) for label in TARGETS[key]['labels']] for key in self.categories]
        self.reset()

    def reset(self):
        self.buffer = np.empty(0, dtype=np.float32)
        self.cursor = 0
        self.total_samples = 0
        self.last_end = 0
        self.finished = False

    def _predict(self, window, valid_samples):
        started = perf_counter()
        padded = np.pad(window, (0, self.window_samples - len(window))).astype(np.float32)
        self.interpreter.set_tensor(self.input['index'], padded.reshape(self.input['shape']))
        self.interpreter.invoke()
        native = self.interpreter.get_tensor(self.output['index']).reshape(-1)
        scores = {key: float(native[indexes].max()) for key, indexes in zip(self.categories, self.groups)}
        top = int(native.argmax())
        end = self.cursor + valid_samples
        self.last_end = end
        return {
            'start': self.cursor / self.sample_rate,
            'end': end / self.sample_rate,
            'available_at': self.total_samples / self.sample_rate,
            'padded': valid_samples < self.window_samples,
            'scores': scores,
            'native_top1': self.labels[top],
            'compute_seconds': perf_counter() - started,
        }

    def feed(self, samples):
        samples = np.asarray(samples)
        if self.finished:
            raise RuntimeError('Call reset() before feeding a new stream after flush().')
        if samples.ndim != 1 or samples.dtype.kind != 'f':
            raise ValueError('Expected one-dimensional floating point PCM at 16 kHz.')
        if not np.isfinite(samples).all() or (samples.size and np.max(np.abs(samples)) > 1.0001):
            raise ValueError('PCM must be finite and normalized to [-1, 1].')
        self.buffer = np.concatenate([self.buffer, samples.astype(np.float32)])
        self.total_samples += len(samples)
        windows = []
        while len(self.buffer) >= self.window_samples:
            windows.append(self._predict(self.buffer[:self.window_samples], self.window_samples))
            self.buffer = self.buffer[self.hop_samples:]
            self.cursor += self.hop_samples
        return windows

    def flush(self):
        """Analyze an uncovered final tail with padding, then close the stream."""
        if self.finished:
            return []
        windows = []
        if self.total_samples > self.last_end and self.buffer.size:
            windows.append(self._predict(self.buffer, len(self.buffer)))
        self.buffer = np.empty(0, dtype=np.float32)
        self.finished = True
        return windows

    def predict_audio(self, samples, chunk_seconds=0.1):
        self.reset()
        chunk_samples = round(chunk_seconds * self.sample_rate)
        if chunk_samples < 1:
            raise ValueError('Chunk duration must be positive.')
        windows = []
        for start in range(0, len(samples), chunk_samples):
            windows.extend(self.feed(samples[start:start + chunk_samples]))
        windows.extend(self.flush())
        return windows


class EventMerger:
    """Hysteresis and temporal grouping, independent for every sound category."""
    def __init__(self, thresholds, off_ratio=0.7, release_windows=2):
        if not 0 < off_ratio <= 1 or release_windows < 1:
            raise ValueError('Invalid hysteresis settings.')
        self.thresholds = dict(thresholds)
        if not self.thresholds or any(not 0 < t <= 1 for t in self.thresholds.values()):
            raise ValueError('Thresholds must be in (0, 1].')
        self.off_ratio = off_ratio
        self.release_windows = release_windows
        self.active = {}
        self.counter = 0

    def update(self, window):
        messages = []
        for category, threshold in self.thresholds.items():
            score = window['scores'][category]
            event = self.active.get(category)
            if event is None:
                if score >= threshold:
                    self.counter += 1
                    event = {
                        'event_id': self.counter, 'category': category,
                        'label': TARGETS[category]['fr'], 'start': window['start'],
                        'end': window['end'], 'first_available_at': window['available_at'],
                        'max_score': score, 'gap': 0,
                    }
                    self.active[category] = event
                    messages.append(self._message(event, 'sound_start', window['available_at']))
            elif score >= threshold * self.off_ratio:
                event['end'] = window['end']
                event['max_score'] = max(event['max_score'], score)
                event['gap'] = 0
            else:
                event['gap'] += 1
                if event['gap'] >= self.release_windows:
                    messages.append(self._message(event, 'sound_end', window['available_at']))
                    del self.active[category]
        return messages

    @staticmethod
    def _message(event, kind, available_at):
        message = {key: value for key, value in event.items() if key != 'gap'}
        message.update(type=kind, emitted_at=available_at)
        return message

    def flush(self, available_at):
        messages = [self._message(event, 'sound_end', available_at) for event in self.active.values()]
        self.active.clear()
        return messages


def merge_windows(windows, thresholds, off_ratio=0.7, release_windows=2):
    merger = EventMerger(thresholds, off_ratio, release_windows)
    messages = []
    for window in windows:
        messages.extend(merger.update(window))
    if windows:
        messages.extend(merger.flush(windows[-1]['available_at']))
    return messages
