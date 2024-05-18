@value
struct constants(AnyType):
    var c: Int
    var pi: Float64

    fn __init__(inout self):
        self.c = 299_792_458
        self.pi = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852110555954930381966446229489

    fn __del__(owned self):
        pass
