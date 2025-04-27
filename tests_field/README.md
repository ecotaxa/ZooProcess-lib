
You will find here tests that need "real" Zooscan projects.

The tests are pytest ones, they all depend on a fixture which reads a .env file, pointing at a directory with real data.

e.g., in `me.env`:
`ZOOSCAN_PROJECTS=/mnt/zooscan_pool/zooscan/remote/complex/piqv/plankton/zooscan_lov/`