"""
engine/server.py — Headless Dedicated Server

Demonstrates the World/Renderer separation.
Runs the full World (terrain gen, physics, block updates) with NO OpenGL.
Exposes a simple TCP protocol for client connections.

Protocol (text-based for simplicity, replace with msgpack for production):
  Client → Server: "MOVE x y z yaw pitch\\n"
  Client → Server: "BREAK bx by bz\\n"
  Client → Server: "PLACE bx by bz block_id\\n"
  Server → Client: "CHUNK cx cz <hex-encoded blocks>\\n"
  Server → Client: "BLOCK bx by bz block_id\\n"
"""

from __future__ import annotations
import socket
import threading
import time
import json
from typing import Dict, Tuple

from world.world import World
from engine.registry import BLOCK_REGISTRY


class DedicatedServer:
    TICK_RATE = 20   # Server ticks per second (like Minecraft)

    def __init__(self, host: str = "0.0.0.0", port: int = 25565,
                  seed: int = 12345, world_name: str = "world"):
        self.host       = host
        self.port       = port
        self.world      = World(seed=seed, world_dir=f"saves/{world_name}")
        self.clients:   Dict[int, dict] = {}   # client_id → {socket, pos, ...}
        self._next_id   = 0
        self._running   = False
        self._lock      = threading.Lock()

    def run(self):
        self._running = True
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_sock.bind((self.host, self.port))
        srv_sock.listen(32)
        srv_sock.settimeout(1.0)
        print(f"[Server] Listening on {self.host}:{self.port}")

        # World tick thread
        tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        tick_thread.start()

        try:
            while self._running:
                try:
                    conn, addr = srv_sock.accept()
                    cid = self._next_id
                    self._next_id += 1
                    print(f"[Server] Player {cid} connected from {addr}")
                    t = threading.Thread(target=self._handle_client,
                                          args=(cid, conn), daemon=True)
                    t.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("[Server] Interrupted")
        finally:
            srv_sock.close()
            self.world.shutdown()

    def _tick_loop(self):
        """20 Hz server tick: update world, broadcast changes."""
        dt = 1.0 / self.TICK_RATE
        last = time.perf_counter()
        while self._running:
            now   = time.perf_counter()
            elapsed = now - last
            if elapsed >= dt:
                last = now
                # Update world for some central position (first player, or 0,0)
                if self.clients:
                    px = list(self.clients.values())[0].get('x', 0)
                    pz = list(self.clients.values())[0].get('z', 0)
                else:
                    px, pz = 0, 0
                self.world.update(px, pz)
            else:
                time.sleep(dt - elapsed)

    def _handle_client(self, cid: int, conn: socket.socket):
        with self._lock:
            self.clients[cid] = {'socket': conn, 'x': 0, 'y': 64, 'z': 0}
        try:
            buf = ""
            while self._running:
                data = conn.recv(1024).decode('utf-8', errors='ignore')
                if not data:
                    break
                buf += data
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    self._process_command(cid, line.strip(), conn)
        except Exception:
            pass
        finally:
            with self._lock:
                del self.clients[cid]
            conn.close()
            print(f"[Server] Player {cid} disconnected")

    def _process_command(self, cid: int, cmd: str, conn: socket.socket):
        parts = cmd.split()
        if not parts:
            return
        verb = parts[0].upper()

        if verb == "MOVE" and len(parts) >= 4:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            with self._lock:
                if cid in self.clients:
                    self.clients[cid].update({'x': x, 'y': y, 'z': z})

        elif verb == "BREAK" and len(parts) >= 4:
            bx, by, bz = int(parts[1]), int(parts[2]), int(parts[3])
            self.world.set_block(bx, by, bz, 0)
            self._broadcast(f"BLOCK {bx} {by} {bz} 0\n")

        elif verb == "PLACE" and len(parts) >= 5:
            bx, by, bz = int(parts[1]), int(parts[2]), int(parts[3])
            block_id   = int(parts[4])
            if block_id in BLOCK_REGISTRY:
                self.world.set_block(bx, by, bz, block_id)
                self._broadcast(f"BLOCK {bx} {by} {bz} {block_id}\n")

        elif verb == "GETCHUNK" and len(parts) >= 3:
            cx, cz = int(parts[1]), int(parts[2])
            chunk = self.world.get_chunk(cx, cz)
            if chunk:
                payload = chunk.blocks.hex()
                try:
                    conn.sendall(f"CHUNK {cx} {cz} {payload}\n".encode())
                except Exception:
                    pass

    def _broadcast(self, msg: str):
        encoded = msg.encode()
        with self._lock:
            dead = []
            for cid, client in self.clients.items():
                try:
                    client['socket'].sendall(encoded)
                except Exception:
                    dead.append(cid)
            for cid in dead:
                del self.clients[cid]
