#!/usr/bin/env python3
import sys
net, gen, hdr = sys.argv[1:4]
with open(net, 'rb') as f:
    data = f.read()
with open(hdr, 'w') as h:
    h.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n")
    h.write(f"extern const unsigned char g_embedded_net[];\n")
    h.write(f"extern const size_t g_embedded_net_size;\n")
    h.write(f"// {len(data)} bytes from {net}\n")
with open(gen, 'w') as g:
    g.write('#include "embedded_net.h"\n')
    g.write(f"const size_t g_embedded_net_size = {len(data)};\n")
    g.write("const unsigned char g_embedded_net[] = {\n")
    for i in range(0, len(data), 12):
        chunk = data[i:i+12]
        g.write(",".join(f"0x{b:02x}" for b in chunk) + ("," if i+12 < len(data) else "") + "\n")
    g.write("};\n")
print(f"Embedded {len(data)} bytes -> {gen}")
