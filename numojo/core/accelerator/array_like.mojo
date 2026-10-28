# ===----------------------------------------------------------------------=== #
# NuMojo: Accelerator array access traits
# Distributed under the Apache 2.0 License with LLVM Exceptions.
# See LICENSE and the LLVM License for more information.
# https://github.com/Mojo-Numerics-and-Algorithms-group/NuMojo/blob/main/LICENSE
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""
Accelerator array access traits (numojo.core.accelerator.array_like).
=====================================================================
Origin-preserving pointer access shared by accelerator-array owners and views.

Exports
-------
- `NDArrayLike`: Common read-only interface for owners and views.
- `NDArrayReadAccess`: Read-access interface returned by `NDArrayLike`.
- `NDArrayWriteAccess`: Mutable access interface for writable arrays and views.
- `NDArrayReadAccessor`: Origin-preserving read pointer and layout.
- `NDArrayWriteAccessor`: Origin-preserving write pointer and layout.
"""

# ===----------------------------------------------------------------------=== #
# NuMojo
# ===----------------------------------------------------------------------=== #
from numojo.core.accelerator.device import Device
from numojo.core.error import NumojoError
from numojo.core.layout.ndlayout import NDLayout, Shape, Strides


trait NDArrayReadAccess(Deinitable, Movable):
    """Read access that safely reborrows accelerator-array storage."""

    comptime ElementType: DType
    comptime DeviceType: Device

    def unsafe_ptr[
        o: Origin
    ](ref[o] self,) -> Pointer[Scalar[Self.ElementType], ImmOrigin(o)]:
        """Return an immutable pointer tied to this accessor borrow.

        Parameters:
            o: Origin of the accessor borrow.

        Returns:
            Immutable pointer whose origin keeps the accessor alive.

        Examples:
            Use `access.unsafe_ptr()` for generic SIMD operations.
        """
        ...

    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied layout metadata.

        Examples:
            Use `access.layout()` to inspect logical strides.
        """
        ...

    def load[
        width: Int
    ](self, index: Int) raises -> SIMD[Self.ElementType, width]:
        """Load SIMD lanes from consecutive logical flat indices.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            index: First logical flat index.

        Returns:
            Loaded SIMD value.

        Raises:
            NumojoError: If the requested lanes are outside the logical extent.

        Examples:
            `access.load[4](0)` loads the first four logical elements.
        """
        ...


trait NDArrayWriteAccess(Deinitable, Movable):
    """Mutable access that safely reborrows accelerator-array storage."""

    comptime ElementType: DType
    comptime DeviceType: Device

    def unsafe_ptr[
        o: MutOrigin
    ](ref[o] self,) -> Pointer[Scalar[Self.ElementType], o]:
        """Return a mutable pointer tied to this accessor borrow.

        Parameters:
            o: Mutable origin of the accessor borrow.

        Returns:
            Mutable pointer whose origin keeps the accessor alive.

        Examples:
            Use `access.unsafe_ptr()` for generic SIMD stores.
        """
        ...

    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied layout metadata.

        Examples:
            Use `access.layout()` to inspect output strides.
        """
        ...


trait NDArrayLike(Sized):
    """Common read-only interface implemented by array owners and views."""

    comptime ElementType: DType
    comptime DeviceType: Device

    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied shape, strides, offset, and size metadata.

        Examples:
            Use `array.layout()` in generic indexing code.
        """
        ...

    def ndim(self) -> Int:
        """Return the logical rank.

        Returns:
            Number of array dimensions.

        Examples:
            Use `array.ndim()` in a generic rank check.
        """
        ...

    def shape(self) -> Shape:
        """Return a copy of the shape.

        Returns:
            Copied logical dimensions.

        Examples:
            Use `array.shape()` when allocating an output owner.
        """
        ...

    def strides(self) -> Strides:
        """Return a copy of the element strides.

        Returns:
            Copied logical strides.

        Examples:
            Use `array.strides()` in generic traversal code.
        """
        ...

    def size(self) -> Int:
        """Return the logical element count.

        Returns:
            Product of the shape dimensions.

        Examples:
            Use `array.size()` to bound a flat traversal.
        """
        ...

    def offset(self) -> Int:
        """Return the first logical element's storage offset.

        Returns:
            Element offset within the backing allocation.

        Examples:
            Use `array.offset()` when adapting a layout.
        """
        ...

    def is_c_contiguous(self) -> Bool:
        """Return whether the logical layout is C contiguous.

        Returns:
            Whether flat C-order pointer traversal is valid.

        Examples:
            Check `array.is_c_contiguous()` before a SIMD pointer loop.
        """
        ...

    def is_f_contiguous(self) -> Bool:
        """Return whether the logical layout is Fortran contiguous.

        Returns:
            Whether flat Fortran-order traversal is valid.

        Examples:
            Use `array.is_f_contiguous()` to select a traversal order.
        """
        ...

    def owns_data(self) -> Bool:
        """Return whether this value owns its backing allocation.

        Returns:
            `True` for an owning array and `False` for a view.

        Examples:
            Use `array.owns_data()` for diagnostics.
        """
        ...

    def is_writeable(self) -> Bool:
        """Return whether this value exposes mutable storage.

        Returns:
            Runtime mirror of the value's write capability.

        Examples:
            Use `array.is_writeable()` for diagnostics and validation.
        """
        ...

    def is_cpu(self) -> Bool:
        """Return whether storage is host-addressable CPU memory.

        Returns:
            `True` for `Device.CPU`.

        Examples:
            Use `array.is_cpu()` before host pointer access.
        """
        ...

    def is_gpu(self) -> Bool:
        """Return whether storage is accelerator memory.

        Returns:
            `True` for a GPU device.

        Examples:
            Use `array.is_gpu()` before launching a kernel.
        """
        ...

    def num_elements(self) -> Int:
        """Return the logical element count.

        Returns:
            Same value as `size()`.

        Examples:
            Use `array.num_elements()` in MAX-compatible adapters.
        """
        ...

    def item(self, flat_index: Int) raises -> Scalar[Self.ElementType]:
        """Return one element at a logical flat index.

        Args:
            flat_index: C-order logical flat index.

        Returns:
            Selected scalar value.

        Raises:
            NumojoError: If the index is invalid or storage is not host-accessible.

        Examples:
            Use `array.item(0)` to read the first logical element.
        """
        ...

    def item(self, *indices: Int) raises -> Scalar[Self.ElementType]:
        """Return one element at N-dimensional coordinates.

        Args:
            indices: One coordinate per axis.

        Returns:
            Selected scalar value.

        Raises:
            NumojoError: If coordinates are invalid or storage is not host-accessible.

        Examples:
            Use `array.item(1, 2)` to read a matrix element.
        """
        ...

    def read_access[
        o: Origin
    ](
        ref[o] self,
    ) -> NDArrayReadAccessor[
        Self.ElementType,
        Self.DeviceType,
        ImmOrigin(o),
    ]:
        """Create read access tied to the underlying storage origin.

        Parameters:
            o: Origin of the current array or view borrow.

        Returns:
            Read accessor whose tracked borrow keeps the storage alive.

        Examples:
            Use `array.read_access()` in generic array algorithms.
        """
        ...


@fieldwise_init
struct NDArrayReadAccessor[
    dtype_: DType,
    device_: Device,
    origin_: ImmOrigin,
](Deinitable, Movable, NDArrayReadAccess):
    """Read-only pointer and layout pair with a storage-derived origin."""

    comptime ElementType = Self.dtype_
    comptime DeviceType = Self.device_
    comptime DataOrigin = Self.origin_

    var _ptr: Pointer[Scalar[Self.ElementType], Self.DataOrigin]
    var _layout: NDLayout

    @always_inline("nodebug")
    def unsafe_ptr[
        o: Origin
    ](ref[o] self,) -> Pointer[Scalar[Self.ElementType], ImmOrigin(o)]:
        """Return a reborrowed pointer to the first logical element.

        Parameters:
            o: Origin of the accessor borrow.

        Returns:
            Immutable pointer tied to this accessor. The accessor's stored
            origin keeps the underlying array or view alive.

        Examples:
            Use `access.unsafe_ptr().unsafe_load[width=4](0)` for SIMD.
        """
        return self._ptr.unsafe_offset(self._layout.offset).unsafe_origin_cast[
            ImmOrigin(o)
        ]()

    @always_inline("nodebug")
    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied layout metadata.

        Examples:
            Use `access.layout().shape()` to inspect dimensions.
        """
        return self._layout.copy()

    @always_inline("nodebug")
    def load[
        width: Int
    ](self, index: Int) raises -> SIMD[Self.ElementType, width]:
        """Load SIMD lanes from consecutive logical flat indices.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            index: First logical flat index.

        Returns:
            Loaded SIMD value.

        Raises:
            NumojoError: If the requested lanes are outside the logical extent.

        Examples:
            `access.load[4](0)` loads the first four logical elements.
        """
        comptime assert Self.DeviceType.type == "cpu", "load is CPU-only"
        if index < 0 or index > self._layout.size - width:
            raise Error(
                NumojoError(
                    category="index",
                    message="SIMD load is outside the logical extent.",
                    location="NDArrayReadAccessor.load",
                )
            )
        if self._layout.is_c_contiguous():
            return self.unsafe_ptr().unsafe_load[width=width](index)
        var result = SIMD[Self.ElementType, width](0)
        comptime for lane in range(width):
            result[lane] = self._ptr[
                unsafe_offset=self._layout.flat_position(index + lane)
            ]
        return result


@fieldwise_init
struct NDArrayWriteAccessor[
    dtype_: DType,
    device_: Device,
    origin_: MutOrigin,
](Deinitable, Movable, NDArrayWriteAccess):
    """Mutable pointer and layout pair with a storage-derived origin."""

    comptime ElementType = Self.dtype_
    comptime DeviceType = Self.device_
    comptime DataOrigin = Self.origin_

    var _ptr: Pointer[Scalar[Self.ElementType], Self.DataOrigin]
    var _layout: NDLayout

    @always_inline("nodebug")
    def unsafe_ptr[
        o: MutOrigin
    ](ref[o] self,) -> Pointer[Scalar[Self.ElementType], o]:
        """Return a mutable reborrow to the first logical element.

        Parameters:
            o: Mutable origin of the accessor borrow.

        Returns:
            Mutable pointer tied to this accessor. The accessor's stored
            origin keeps the underlying array or view alive.

        Examples:
            Use `access.unsafe_ptr().unsafe_store[width=4](0, value)`.
        """
        return self._ptr.unsafe_offset(self._layout.offset).unsafe_origin_cast[
            o
        ]()

    @always_inline("nodebug")
    def layout(self) -> NDLayout:
        """Return a copy of the logical layout.

        Returns:
            Copied layout metadata.

        Examples:
            Use `access.layout().strides()` to inspect strides.
        """
        return self._layout.copy()

    @always_inline("nodebug")
    def store[
        width: Int
    ](mut self, index: Int, value: SIMD[Self.ElementType, width],) raises:
        """Store SIMD lanes to consecutive logical flat indices.

        Parameters:
            width: Number of SIMD lanes.

        Args:
            index: First logical flat index.
            value: SIMD value to store.

        Raises:
            NumojoError: If the requested lanes are outside the logical extent.

        Examples:
            `access.store[4](0, value)` stores four logical elements.
        """
        comptime assert Self.DeviceType.type == "cpu", "store is CPU-only"
        if index < 0 or index > self._layout.size - width:
            raise Error(
                NumojoError(
                    category="index",
                    message="SIMD store is outside the logical extent.",
                    location="NDArrayWriteAccessor.store",
                )
            )
        if self._layout.is_c_contiguous():
            self.unsafe_ptr().unsafe_store[width=width](index, value)
            return
        comptime for lane in range(width):
            self._ptr[
                unsafe_offset=self._layout.flat_position(index + lane)
            ] = value[lane]
