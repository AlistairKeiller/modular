# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Regression test for `matmul[transpose_b=True]` with N == 1 and M > 1.

`gemv_gpu_dispatch`'s scalar `GemvKernel` branch used the transposed launch
(A and B swapped, one warp per column of C) for every `transpose_b` product.
That launch is the M == 1 case; with N == 1 and M > 1 it ran a single warp and
wrote only c[0].
"""

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu
from std.testing import assert_equal


def test[
    dtype: DType, M: Int, N: Int, K: Int, transpose_b: Bool
](ctx: DeviceContext) raises:
    """c[M, N] = a[M, K] @ b (or a @ b^T). `a` is all ones and column j of
    `b` (row j when transposed) holds j + 1, so every c[i, j] must equal
    K * (j + 1); the values are small integers, exact in every dtype."""
    print(M, "x", N, "x", K, "transpose_b", transpose_b, dtype)

    var a_host = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_host = ctx.enqueue_create_host_buffer[dtype](M * N)
    for i in range(M * K):
        a_host[i] = 1
    for k in range(K):
        for j in range(N):
            b_host[j * K + k if transpose_b else k * N + j] = Scalar[dtype](
                j + 1
            )
    for i in range(M * N):
        # Sentinel: a row the kernel never writes keeps it and fails below.
        c_host[i] = -1

    var a_dev = ctx.enqueue_create_buffer[dtype](M * K)
    var b_dev = ctx.enqueue_create_buffer[dtype](K * N)
    var c_dev = ctx.enqueue_create_buffer[dtype](M * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)

    comptime b_dim0 = N if transpose_b else K
    comptime b_dim1 = K if transpose_b else N
    var a_tensor = TileTensor(a_dev, row_major(Coord(M, Idx[K])))
    var b_tensor = TileTensor(b_dev, row_major(Coord(Idx[b_dim0], Idx[b_dim1])))
    var c_tensor = TileTensor(c_dev, row_major(Coord(M, Idx[N])))

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_tensor, a_tensor.as_imm(), b_tensor.as_imm(), ctx
    )
    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    for i in range(M):
        for j in range(N):
            assert_equal(
                c_host[i * N + j].cast[.float32](),
                Float32(K * (j + 1)),
                String("c[", i, ", ", j, "]"),
            )


def main() raises:
    with DeviceContext() as ctx:
        comptime K = 1024
        # float32 with N == 1 takes the scalar `GemvKernel` path, the one that
        # wrote only c[0] for the transposed layout. M == 1 covers the row
        # case the transposed launch is actually for.
        comptime for m in range(1, 5):
            test[.float32, m, 1, K, False](ctx)
            test[.float32, m, 1, K, True](ctx)
        test[.float32, 17, 1, K, False](ctx)
        test[.float32, 17, 1, K, True](ctx)
        test[.float32, 1, 3, K, True](ctx)
        # bfloat16 with K % simd_width == 0 takes `GemvKernelVector`; one
        # shape per layout guards that path.
        test[.bfloat16, 3, 1, K, True](ctx)
        test[.bfloat16, 1, 3, K, True](ctx)
