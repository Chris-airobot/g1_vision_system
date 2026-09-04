L = 0.40  # length in meters
W = 0.30  # width in meters
H = 0.30  # height in meters

x = L / 2
y = W / 2
z = H / 2

vertices = [
    (-x, -y, -z),
    ( x, -y, -z),
    ( x,  y, -z),
    (-x,  y, -z),
    (-x, -y,  z),
    ( x, -y,  z),
    ( x,  y,  z),
    (-x,  y,  z),
]

faces = [
    (1, 2, 3), (1, 3, 4),  # bottom
    (5, 8, 7), (5, 7, 6),  # top
    (1, 5, 6), (1, 6, 2),  # front
    (2, 6, 7), (2, 7, 3),  # right
    (3, 7, 8), (3, 8, 4),  # back
    (4, 8, 5), (4, 5, 1),  # left
]

output_file = "box.obj"

with open(output_file, "w") as f:
    f.write("# 40 cm x 30 cm x 30 cm box\n")
    f.write("# Units: meters\n\n")

    for vx, vy, vz in vertices:
        f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")

    f.write("\n")

    for a, b, c in faces:
        f.write(f"f {a} {b} {c}\n")

print(f"Created {output_file}")
print(f"Dimensions: {L} m x {W} m x {H} m")
print("Origin is at the center of the box.")