# project-107cf1

A voxel game engine and sandbox built from scratch in Rust with a custom Vulkan renderer.

## Features

### Rendering Engine
- Custom Vulkan 1.3 rendering backend with modern GPU-driven techniques
- Deferred shading with G-buffer
- Bloom, tonemapping, and post-processing pipeline
- GPU frustum culling and depth pyramid occlusion
- MSDF-based text rendering
- Basis Universal texture compression

### UI System
- Declarative reactive UI framework with state management
- Flexbox-style layout engine (flow, alignment, padding, constraints)
- HUD overlay with health/satiety indicators
- Inventory system with item slots
- Main menu with world selection
- Debug overlay with runtime information

### World System
- Infinite procedural terrain generation using layered noise functions
- Temperature & humidity maps for implicit biomes
- Block-based voxel world with 32³ cluster chunks
- Dynamic lighting with light propagation
- Liquid simulation with flow dynamics
- Structure generation system (caves, buildings, etc.)

### Physics & Gameplay
- AABB collision detection
- Gravity and player movement physics
- First-person camera controls

### Architecture
- Multi-threaded task execution with work-stealing
- Async coroutine system
- Entity Component System (ECS)
- Resource caching and streaming
- Cross-platform support (macOS, Windows, Linux)

## Project Structure

```
├── base/           # Core game logic (world, physics, registry)
├── client/         # Application entry point and game client
│   ├── engine/     # Rendering engine
│   │   ├── vk-wrapper/   # Vulkan abstraction layer
│   │   ├── msdfgen/      # MSDF font generation
│   │   └── shaders/      # Engine shaders
│   ├── res/        # Game resources (textures, models, fonts)
│   └── src/        # Client source (rendering, UI, game logic)
└── common/         # Shared utilities and types
```

## Building

### Prerequisites

- Rust (latest stable)
- VulkanSDK 1.3+
- LLVM/Clang

### Platform-specific Dependencies

**Windows**
```
- Visual Studio 2019+
- LLVM
```

**Ubuntu/Debian**
```bash
sudo apt install build-essential xorg-dev llvm-dev libclang-dev clang
```

**macOS**
```bash
brew install llvm
```

### Build & Run

```bash
cd client/work_dir
cargo build --release
cargo run --release
```

## License

See [LICENSE](LICENSE) for details.
