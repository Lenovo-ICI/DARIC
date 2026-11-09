from .kvcache import get_partition_cache_id
from ..configs import ProcessLocalConfig
from ..logging import get_logger
from ..management.service.zmq_rpc import ZMQRPCClient, ZMQRPCProxy

logger = get_logger(__name__)


class LookupClient:
    def __init__(
            self,
            host: str = "127.0.0.1" ,
            port: int = 50000,
            layerwise: bool = False
        ):
        self._rpc_client = ZMQRPCClient(host, port, timeout=30.0)
        self.storage_mgr = ZMQRPCProxy(self._rpc_client, "store")
        if layerwise and ProcessLocalConfig.pp_size > 1:
            raise ValueError("Layerwise lookup is not supported when pp_size > 1.")
        
        self.num_layers = len(ProcessLocalConfig.cache_topo.get_all_layer_names())
        self.layerwise = layerwise
        self.tp_size = ProcessLocalConfig.tp_size
        self.world_size = ProcessLocalConfig.world_size
        self.is_deepseek_mla = ProcessLocalConfig.is_deepseek_mla
        self.partitioned = bool(
            ProcessLocalConfig.cache_topo.get_mamba_group_idxs()
        )

    def lookup(self, keys):
        if len(keys) == 0:
            return []
        if self.is_deepseek_mla:
            ranks = [ProcessLocalConfig.get_local_mla_saver_rank()]
        else:
            ranks = list(range(self.world_size))
        layers = range(self.num_layers) if self.layerwise else (None,)
        if self.layerwise:
            suffixes = [f"_{r}_{l}" for r in ranks for l in layers]
        else:
            suffixes = [f"_{r}" for r in ranks]
        stride = len(suffixes)
        if self.partitioned:
            partitions = ProcessLocalConfig.cache_topo.get_cache_partitions()
            physical_keys = [
                get_partition_cache_id(key, partition)
                for key in keys
                for partition, _ in partitions
            ]
            stride *= len(partitions)
        else:
            physical_keys = keys
        lookup_keys = [key + s for key in physical_keys for s in suffixes]
        results_ = self.storage_mgr.lookup(lookup_keys)
        return [
            all(results_[i:i + stride])
            for i in range(0, len(results_), stride)
        ]

    def close(self):
        if hasattr(self, '_rpc_client'):
            self._rpc_client.close()