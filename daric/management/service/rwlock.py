import threading


class _RWLock:
    def __init__(self):
        self._cond = threading.Condition(threading.RLock())
        self._readers = 0
        self._writer_active = False
        self._writers_waiting = 0

    def acquire_read(self):
        with self._cond:
            while self._writer_active or self._writers_waiting > 0: self._cond.wait()
            self._readers += 1

    def release_read(self):
        with self._cond:
            self._readers -= 1
            if self._readers == 0: self._cond.notify_all()

    def acquire_write(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writer_active or self._readers > 0: self._cond.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self):
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    class _ReadCtx:
        def __init__(self, lock): 
            self._lock = lock
        def __enter__(self): 
            self._lock.acquire_read(); return self
        def __exit__(self, *args): 
            self._lock.release_read()

    class _WriteCtx:
        def __init__(self, lock): self._lock = lock
        def __enter__(self): self._lock.acquire_write(); return self
        def __exit__(self, *args): self._lock.release_write()

    def read_lock(self):
        return _RWLock._ReadCtx(self)
    def write_lock(self):
        return _RWLock._WriteCtx(self)