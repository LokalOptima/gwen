"""
Client for GWEN inference server's batch endpoints.

Extracted from train_mtp.py for reuse across pipeline scripts.
"""

import http.client
import json
import struct
import sys

import numpy as np
import torch


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class GwenClient:
    """Client for GWEN inference server's batch hidden state extraction."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.conn = http.client.HTTPConnection(host, port, timeout=300)
        self._check_health()

    def _check_health(self):
        try:
            self.conn.request("GET", "/health")
            resp = self.conn.getresponse()
            data = json.loads(resp.read().decode())
            log(f"GWEN server: {data.get('model', '?')}, "
                f"n_embed={data.get('n_embed', '?')}, "
                f"max_seq={data.get('max_seq', '?')}")
        except ConnectionRefusedError:
            raise RuntimeError(
                f"Cannot connect to GWEN server at {self.host}:{self.port}. "
                f"Start it first: build/gwen_dev_server --model <path.gguf> --port {self.port}"
            )

    def _reconnect(self):
        self.conn.close()
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=300)

    def _post(self, path, body):
        headers = {"Content-Type": "application/octet-stream"}
        try:
            self.conn.request("POST", path, body=body, headers=headers)
            resp = self.conn.getresponse()
        except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
            self._reconnect()
            self.conn.request("POST", path, body=body, headers=headers)
            resp = self.conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"GWEN server error {resp.status}: {resp.read().decode()}")
        return resp.read()

    def batch_extract_with_preds(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract hidden states AND main model predictions.

        Returns: (hidden [B, L, n_embed] float16, predictions [B, L] int32)
        """
        token_np = token_ids.cpu().numpy().astype(np.int32)
        B, L = token_np.shape
        body = struct.pack('<II', B, L) + token_np.tobytes()
        data = self._post("/batch_extract?preds=1", body)
        B2, L2, d = struct.unpack('<III', data[:12])
        hidden_bytes = B2 * L2 * d * 2
        hidden = np.frombuffer(data[12:12 + hidden_bytes], dtype=np.float16).reshape(B2, L2, d).copy()
        preds = np.frombuffer(data[12 + hidden_bytes:], dtype=np.int32).reshape(B2, L2).copy()
        return torch.from_numpy(hidden), torch.from_numpy(preds)

    def batch_logits(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract hidden states AND teacher logits over restricted vocab.

        Calls the dev_server's /batch_logits endpoint.
        Returns: (hidden [B, L, n_embed] float16, teacher_logits [B, L, K] float16)
        """
        token_np = token_ids.cpu().numpy().astype(np.int32)
        B, L = token_np.shape
        body = struct.pack('<II', B, L) + token_np.tobytes()
        data = self._post("/batch_logits", body)
        B2, L2, K = struct.unpack('<III', data[:12])
        N = B2 * L2
        # Derive n_embed from total data size:
        # total = N * n_embed * 2 + N * K * 2 = N * 2 * (n_embed + K)
        total_data = len(data) - 12
        n_embed = total_data // (N * 2) - K
        assert n_embed == 1024, f"Expected n_embed=1024, got {n_embed} (data={total_data}, N={N}, K={K})"
        hidden_bytes = N * n_embed * 2
        logits_bytes = N * K * 2
        hidden = np.frombuffer(data[12:12 + hidden_bytes], dtype=np.float16).reshape(B2, L2, n_embed).copy()
        logits = np.frombuffer(data[12 + hidden_bytes:12 + hidden_bytes + logits_bytes],
                               dtype=np.float16).reshape(B2, L2, K).copy()
        return torch.from_numpy(hidden), torch.from_numpy(logits)

    def batch_logits_with_p_idk(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract hidden states, teacher logits, AND p_idk from dev_server.

        Calls /batch_logits?p_idk=1. Requires dev_server built with p_idk support.
        Returns: (hidden [B, L, 1024] F16, logits [B, L, K] F16, p_idk [B, L] F32)
        """
        token_np = token_ids.cpu().numpy().astype(np.int32)
        B, L = token_np.shape
        body = struct.pack('<II', B, L) + token_np.tobytes()
        data = self._post("/batch_logits?p_idk=1", body)
        B2, L2, K = struct.unpack('<III', data[:12])
        N = B2 * L2
        off = 12
        hidden_bytes = N * 1024 * 2
        hidden = np.frombuffer(data[off:off + hidden_bytes], dtype=np.float16).reshape(B2, L2, 1024).copy()
        off += hidden_bytes
        logits_bytes = N * K * 2
        logits = np.frombuffer(data[off:off + logits_bytes], dtype=np.float16).reshape(B2, L2, K).copy()
        off += logits_bytes
        pidk_bytes = N * 4
        p_idk = np.frombuffer(data[off:off + pidk_bytes], dtype=np.float32).reshape(B2, L2).copy()
        return torch.from_numpy(hidden), torch.from_numpy(logits), torch.from_numpy(p_idk)
