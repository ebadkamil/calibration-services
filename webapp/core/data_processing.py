from concurrent.futures import ThreadPoolExecutor
import functools

import queue
from threading import Thread, Event
import numpy as np

from .config import config


class ProcessedData:
    def __init__(self, tid):
        self._tid = tid
        self.image = None

    @property
    def tid(self):
        return self._tid


class DataProcessorWorker(Thread):
    def __init__(self, out_queue):
        super().__init__()

        self._running = False
        self._out_queue = out_queue
        self._fitting_type = None
        self._geom_file = None
        self._geom = None
        self._source_name = None
        self._raw_data = None

    def run(self):
        self._running = True

        proc_data = ProcessedData(tid)

        while self._running:
            if self._raw_data is not None:
                try:
                    self._process(self._raw_data, proc_data)
                except Exception as ex:
                    self._raw_data = None
                self.mask_image(assembled, threshold_mask=threshold_mask)
                mean_image = np.mean(assembled, axis=0)

                while self._running:
                    try:
                        self._out_queue.put(
                            proc_data, timeout=config["TIME_OUT"])
                        self._raw_data = None
                        break
                    except queue.Full:
                        continue
            else:
                continue

    def _process(self, data, processed):
        pass

    def onRawDataAvailable(self, data):
        self._data = data

    def mask_image(self, image, threshold_mask=None):

        def parallel(i):
            image[i][np.isnan(image[i])] = 0
            if threshold_mask is not None:
                a_min, a_max = threshold_mask
                np.clip(image[i], a_min, a_max, out=image[i])

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(parallel, range(image.shape[0]))

    def onAnalysisTypeChange(self, value):
        if self._fitting_type != value:
            self._fitting_type = value

    def onSourceNameChange(self, value):
        self._source_name = value

    def terminate(self):
        self._running = False


