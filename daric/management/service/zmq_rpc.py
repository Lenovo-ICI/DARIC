import pickle
import threading
import time
from typing import Any, Dict, Optional

import zmq

from ...logging import get_logger

logger = get_logger(__name__)

# Sentinel target / method name used by the lightweight server-liveness probe.
_PING_TARGET = "__ping__"
_PING_METHOD = "__ping__"


class ZMQRPCServer:
    def __init__(
        self,
        host: str,
        port: int,
        targets: Dict[str, Any],
        num_workers: int = 16,
        context: Optional[zmq.Context] = None,
    ):
        self.host = host
        self.port = port
        self.targets = targets
        self.num_workers = num_workers
        self.context = context or zmq.Context.instance()
        self._stop = False
        self._router: Optional[zmq.Socket] = None
        self._dealer: Optional[zmq.Socket] = None
        self._worker_threads: list = []
        self._proxy_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Bind sockets and launch the proxy + worker threads (non-blocking)."""
        self._router = self.context.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.bind(f"tcp://{self.host}:{self.port}")

        self._dealer = self.context.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.LINGER, 0)
        self._dealer.bind("inproc://zmq-rpc-workers")

        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"zmq-rpc-worker-{i}",
                daemon=True,
            )
            t.start()
            self._worker_threads.append(t)

        self._proxy_thread = threading.Thread(
            target=self._proxy_loop,
            name="zmq-rpc-proxy",
            daemon=True,
        )
        self._proxy_thread.start()

    def stop(self) -> None:
        """Signal all threads to stop and close the frontend sockets."""
        self._stop = True
        if self._router is not None:
            try:
                self._router.close(linger=0)
            except Exception:
                pass
        if self._dealer is not None:
            try:
                self._dealer.close(linger=0)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # internal
    # ------------------------------------------------------------------ #
    def _proxy_loop(self) -> None:
        try:
            zmq.proxy(self._router, self._dealer)
        except zmq.ContextTerminated:
            pass
        except zmq.ZMQError as exc:
            if not self._stop:
                logger.error("ZMQ proxy terminated with error: %s", exc)

    def _worker_loop(self) -> None:
        socket = self.context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect("inproc://zmq-rpc-workers")
        while not self._stop:
            try:
                payload = socket.recv()
            except zmq.ZMQError:
                break
            response = self._handle_payload(payload)
            try:
                socket.send(response)
            except zmq.ZMQError:
                break
        socket.close(linger=0)

    def _handle_payload(self, payload: bytes) -> bytes:
        try:
            target_name, method_name, args, kwargs = pickle.loads(payload)
        except Exception as exc:
            logger.exception("ZMQ RPC: failed to unpickle request")
            return pickle.dumps(("error", exc))

        # Lightweight liveness probe used by the client connection helper.
        if method_name == _PING_METHOD:
            return pickle.dumps(("ok", None))

        try:
            target = self.targets.get(target_name)
            if target is None:
                raise ValueError(f"Unknown RPC target '{target_name}'")
            method = getattr(target, method_name, None)
            if method is None or not callable(method):
                raise AttributeError(
                    f"Target '{target_name}' has no callable method '{method_name}'"
                )
            result = method(*args, **kwargs)
            return pickle.dumps(("ok", result))
        except Exception as exc:
            logger.exception(
                "ZMQ RPC error: %s.%s args=%s kwargs=%s",
                target_name, method_name, args, kwargs,
            )
            return pickle.dumps(("error", exc))


class ZMQRPCClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 30.0,
        connect_timeout: float = 30.0,
        connect_interval: float = 0.5,
        context: Optional[zmq.Context] = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.connect_interval = connect_interval
        self.context = context or zmq.Context.instance()
        self._socket: Optional[zmq.Socket] = None
        self._lock = threading.Lock()
        self._connect()

    # ------------------------------------------------------------------ #
    # internal socket management
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
        sock = self.context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        sock.connect(f"tcp://{self.host}:{self.port}")
        self._socket = sock

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def call(self, target_name: str, method_name: str, *args, timeout: Optional[float] = None, **kwargs) -> Any:
        payload = pickle.dumps((target_name, method_name, args, kwargs))
        call_timeout = self.timeout if timeout is None else timeout
        # Apply the per-call timeout by temporarily switching RCVTIMEO.
        # ZMQ RCVTIMEO semantics: -1 = block forever, 0 = non-blocking,
        # >0 = timeout in ms. Map inf/very-large to -1 (block forever).
        if call_timeout == float('inf') or call_timeout >= 3.156e7:  # ~1 year
            rcvtimeo_ms = -1
        else:
            rcvtimeo_ms = int(call_timeout * 1000)
        with self._lock:
            self._socket.setsockopt(zmq.RCVTIMEO, rcvtimeo_ms)
            try:
                self._socket.send(payload)
                response = self._socket.recv()
            except zmq.ZMQError as exc:
                if exc.errno == zmq.EAGAIN:
                    self._connect()
                    raise TimeoutError(
                        f"ZMQ RPC call {target_name}.{method_name} timed out "
                        f"after {call_timeout:.1f}s ({self.host}:{self.port})"
                    )
                raise
            status, result = pickle.loads(response)
            if status == "ok":
                return result
            raise result

    def wait_for_server(
        self,
        timeout: Optional[float] = None,
        interval: Optional[float] = None,
    ) -> None:
        """Block until the server responds to a ping (or raise ``TimeoutError``)."""
        timeout = self.connect_timeout if timeout is None else timeout
        interval = self.connect_interval if interval is None else interval
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.call(_PING_TARGET, _PING_METHOD)
                return
            except (TimeoutError, zmq.ZMQError, OSError):
                time.sleep(interval)
        raise TimeoutError(
            f"Could not reach ZMQ RPC server at {self.host}:{self.port} "
            f"within {timeout:.1f}s"
        )

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close(linger=0)
                except Exception:
                    pass
                self._socket = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ZMQRPCProxy:
    def __init__(self, client: ZMQRPCClient, target_name: str):
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_target_name", target_name)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        client = object.__getattribute__(self, "_client")
        target_name = object.__getattribute__(self, "_target_name")

        def _remote_method(*args, **kwargs):
            return client.call(target_name, name, *args, **kwargs)

        _remote_method.__name__ = name
        return _remote_method