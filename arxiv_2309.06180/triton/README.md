tl.program_id(0)  = which block/program am I?
tl.arange(...)    = offsets inside this block
tl.load(...)      = read from GPU memory
tl.store(...)     = write to GPU memory
mask              = avoid reading past the end
BLOCK_SIZE        = how many elements one program handles

@triton.jit       = compile this python function into a GPU kernel
 jit(just in time compiler)

offsets           = exact positions triton program is gonna work on
 block_size = 4
 pid = 2 
 block_start = 2 * 4 (pid * block_size)
 offsets = block_start + tl.arange(0, block_size) [8 + 0, 8 + 1, 8 + 2, 8 + 3] = [8, 9, 10, 11] 

mask              = prevents reading/writing outise the tensor
 n_elements = 10
 offsets = [8, 9, 10, 11]
 mask = offsets < n_elements
 mask = [True, True, False, False]

memory_indexing   = calculating where to read/write in GPU mem
