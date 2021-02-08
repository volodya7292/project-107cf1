# project-107cf1 - A Specification

## 1. World
A World is a system of clusters that allow dynamically modifying its contents.
A Cluster in a world may have different LOD (level of detail).
LOD is expressed in cluster scale. Default scale is 1.
Contents of a cluster can be modified if and only if its scale is 1.

### 1.1. Cluster
A Cluster is a 64x64x64x4 (X, Y, Z, layers) grid of matter points.
Each XYZ point may have multiple layers (up to 4) to allow blending different materials
(e.g. water on solid surface).

```rust
struct MatterPoint {
    density: u8,
    material: u16,
}
```

- `density` is an isosurface value.
- `material` is the material of this point.

Matter point is visible if `density > 127`.

### 1.2. Cluster triangulation
To triangulate clusters smoothly, boundaries between them must also be triangulated.
Triangulation of layer **L** of cluster **C** consists of the following steps.

1. Collect all nodes **N** of 64x64x64 grid of layer **L** of cluster **C**.
2. Find all adjacent clusters **J** to **C**.
3. Collect boundary nodes **B** of **J**.
4. Create grid **BG** and fill it with boundary matter points of **B**.
   Fill holes in **BG** to compensate different cluster scales.
   Create nodes **BN** from grid **BG**.
5. Create an octree **O** and insert in it **N** and **BN** nodes.
6. Triangulate **O** using dual-contouring or similar technique.
   
A cluster node is a 2x2x2 structure of matter points.
It is useful in constructing cluster octree with boundary nodes.
