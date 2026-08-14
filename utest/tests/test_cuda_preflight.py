from __future__ import annotations

from types import SimpleNamespace

from utest.cuda_preflight import run_preflight


class _Tensor:
    def to(self, device):
        assert device == "cuda"
        return self

    def __matmul__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 1024.0


class _Cuda:
    def is_available(self):
        return True

    def device_count(self):
        return 1

    def current_device(self):
        return 0

    def get_device_name(self, index):
        return "test GPU"

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=80 * 1024**3)

    def synchronize(self):
        return None

    def mem_get_info(self, index):
        return 70 * 1024**3, 80 * 1024**3


class _Torch:
    __version__ = "2.7.1+cu128"
    version = SimpleNamespace(cuda="12.8")
    cuda = _Cuda()
    bfloat16 = "bfloat16"

    def ones(self, shape, *, dtype):
        assert dtype == self.bfloat16
        return _Tensor()


def test_cuda_preflight_records_a_real_kernel_smoke() -> None:
    report = run_preflight(_Torch(), driver_version="525.105.17")
    assert report["status"] == "passed"
    assert report["device"]["name"] == "test GPU"
    assert report["device"]["free_memory_gb"] == 70.0
    assert report["runtime"] == {
        "torch": "2.7.1+cu128",
        "torch_cuda": "12.8",
        "driver": "525.105.17",
    }
    assert report["smoke"]["operation"] == "cpu_to_cuda_bf16_matmul"
    assert report["smoke"]["checksum"] == 1024.0


def test_cuda_preflight_turns_cuda_exception_into_evidence() -> None:
    class BrokenTorch(_Torch):
        def ones(self, shape, *, dtype):
            raise RuntimeError("CUDA driver error: invalid argument")

    report = run_preflight(BrokenTorch(), driver_version="525.105.17")
    assert report["status"] == "failed"
    assert report["error"] == "RuntimeError: CUDA driver error: invalid argument"
