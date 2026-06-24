tl.program_id(0)  = which block/program am I?
tl.arange(...)    = offsets inside this block
tl.load(...)      = read from GPU memory
tl.store(...)     = write to GPU memory
mask              = avoid reading past the end
BLOCK_SIZE        = how many elements one program handles