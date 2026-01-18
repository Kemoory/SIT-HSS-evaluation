import torch
import numpy as np
import math
from numba import njit, prange
from skimage.color import rgb2lab

class SITHSS:
    def __init__(self, n_segments=200, t=0.1, tau=2e-7, max_radius=5, device=None):
        self.K = int(n_segments)
        self.t = float(t)
        self.tau = float(tau)
        self.max_radius = int(max_radius)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shifts = self._generate_shifts()

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _find_best_neighbors(active_nodes, adj_v, adj_w, adj_offsets, vol, cut, LOG_VG, INV_VG, tau, force_merge=False):
        n_active = len(active_nodes)
        best_v = np.full(n_active, -1, dtype=np.int64)
        
        for i in prange(n_active):
            u = active_nodes[i]
            start, end = adj_offsets[u], adj_offsets[u+1]
            if start == end: continue
            
            max_d = -1e18
            b_v = -1
            vol_u, cut_u = vol[u], cut[u]
            term_u = max(0.0, vol_u - cut_u) * math.log(vol_u + 1e-20)
            
            for j in range(start, end):
                v, w_uv = adj_v[j], adj_w[j]
                v_new = vol_u + vol[v]
                g_new = cut_u + cut[v] - 2 * w_uv
                
                term_v = max(0.0, vol[v] - cut[v]) * math.log(vol[v] + 1e-20)
                term_new = max(0.0, v_new - g_new) * math.log(v_new + 1e-20)
                
                delta = (term_u + term_v - term_new + (2 * w_uv * LOG_VG)) * INV_VG
                
                if force_merge:
                    if delta > max_d:
                        max_d, b_v = delta, v
                else:
                    if delta > tau and delta > max_d:
                        max_d, b_v = delta, v
            
            best_v[i] = b_v
        return best_v

    def _get_features(self, image):
        H, W = image.shape[:2]
        img_lab = rgb2lab(image).astype(np.float32)
        img_lab[:,:,0] /= 100.0
        img_lab[:,:,1:] /= 128.0
        
        tensor_lab = torch.from_numpy(img_lab).permute(2, 0, 1).to(self.device)
        y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        
        scale = max(H, W)
        pos = torch.stack([x.float() / scale, y.float() / scale], dim=0)
        return torch.cat([tensor_lab, pos], dim=0), H, W

    def _get_shifts_at_radius(self, r):
        shifts = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):

                dist = math.sqrt(dy**2 + dx**2)
                if r - 1 < dist <= r:
                    shifts.append((dy, dx))
        return shifts

    def _compute_initial_graph(self, features, H, W):
        n_pixels = H * W
        feat_c = features[:3].view(3, n_pixels)
        feat_s = features[3:].view(2, n_pixels)
        idx = torch.arange(n_pixels, device=self.device).view(H, W)
        
        all_u, all_v, all_w = [], [], []
        
        degrees = torch.zeros(n_pixels, device=self.device)
        prev_h1 = -1e18
        
        print(f"--- Recherche du rayon optimal (tau={self.tau}) ---")

        for r in range(1, self.max_radius + 1):
            current_shifts = self._get_shifts_at_radius(r)
            if not current_shifts: continue
            
            for dy, dx in current_shifts:
                y_s, y_e = max(0, dy), min(H, H + dy)
                x_s, x_e = max(0, dx), min(W, W + dx)
                yo_s, yo_e = max(0, -dy), min(H, H - dy)
                xo_s, xo_e = max(0, -dx), min(W, W - dx)
                
                u_i = idx[yo_s:yo_e, xo_s:xo_e].reshape(-1)
                v_i = idx[y_s:y_e, x_s:x_e].reshape(-1)
                
                dist_c = torch.sum((feat_c[:, u_i] - feat_c[:, v_i])**2, dim=0)
                dist_s = torch.sqrt(torch.sum((feat_s[:, u_i] - feat_s[:, v_i])**2, dim=0) + 1e-12)
                rhos = dist_c * dist_s
                
                w = torch.exp(-rhos / (self.t * rhos.mean() + 1e-12))
                
                degrees.index_add_(0, u_i, w)
                
                all_u.append(u_i.cpu())
                all_v.append(v_i.cpu())
                all_w.append(w.cpu())

            vol_g = degrees.sum()
            p = degrees / (vol_g + 1e-20)
            h1_r = -torch.sum(p * torch.log(p + 1e-20)).item()
            
            delta_h1 = h1_r - prev_h1
            print(f" Rayon {r}: H1 = {h1_r:.4f}, Delta = {delta_h1:.6e}")

            if r > 1 and delta_h1 < self.tau:
                print(f" -> Seuil atteint. Arrêt au rayon {r}")
                break
                
            prev_h1 = h1_r

        del degrees, feat_c, feat_s, idx
        torch.cuda.empty_cache()

        return torch.cat(all_u).numpy(), torch.cat(all_v).numpy(), torch.cat(all_w).numpy()

    def fit(self, image):
        features, H, W = self._get_features(image)
        u_idx, v_idx, weights = self._compute_initial_graph(features, H, W)
        
        n_pixels = H * W
        adj = [{} for _ in range(n_pixels)]
        vol = np.zeros(n_pixels, dtype=np.float64)
        for i in range(len(u_idx)):
            u, v, w = int(u_idx[i]), int(v_idx[i]), float(weights[i])
            adj[u][v] = w
            vol[u] += w

        cut = vol.copy()
        active_regions = set(range(n_pixels))
        self.parent = np.arange(n_pixels, dtype=np.int64)
        V_G = vol.sum()
        LOG_VG = math.log(V_G + 1e-20)
        INV_VG = 1.0 / V_G

        while len(active_regions) > self.K:
            nodes_sorted = np.array(list(active_regions), dtype=np.int64)
            
            adj_v_flat, adj_w_flat = [], []
            adj_offsets = np.zeros(n_pixels + 1, dtype=np.int64)
            curr = 0
            for i in range(n_pixels):
                adj_offsets[i] = curr
                for v_n, w_n in adj[i].items():
                    adj_v_flat.append(v_n); adj_w_flat.append(w_n)
                    curr += 1
            adj_offsets[n_pixels] = curr
            
            best_v_arr = self._find_best_neighbors(
                nodes_sorted, np.array(adj_v_flat, dtype=np.int64), 
                np.array(adj_w_flat, dtype=np.float64), adj_offsets, 
                vol, cut, LOG_VG, INV_VG, self.tau, force_merge=False
            )
            
            best_neighbor = {nodes_sorted[i]: best_v_arr[i] for i in range(len(nodes_sorted)) if best_v_arr[i] != -1}

            if not best_neighbor and len(active_regions) > self.K:
                best_v_arr = self._find_best_neighbors(
                    nodes_sorted, np.array(adj_v_flat, dtype=np.int64), 
                    np.array(adj_w_flat, dtype=np.float64), adj_offsets, 
                    vol, cut, LOG_VG, INV_VG, self.tau, force_merge=True
                )
                best_neighbor = {nodes_sorted[i]: best_v_arr[i] for i in range(len(nodes_sorted)) if best_v_arr[i] != -1}

            if not best_neighbor: break 

            processed = set()
            for u in list(best_neighbor.keys()):
                if u in processed or u not in active_regions: continue
                v = best_neighbor[u]
                if v != -1 and best_neighbor.get(v) == u and v in active_regions and v not in processed:
                    if len(active_regions) <= self.K: break
                    
                    w_uv = adj[u][v]
                    vol[u] += vol[v]
                    cut[u] = cut[u] + cut[v] - 2 * w_uv
                    
                    for neighbor, w_vn in list(adj[v].items()):
                        if neighbor == u: continue
                        new_w = adj[u].get(neighbor, 0.0) + w_vn
                        adj[u][neighbor] = adj[neighbor][u] = new_w
                        if v in adj[neighbor]: del adj[neighbor][v]
                    
                    adj[v] = {}; adj[u].pop(v, None)
                    active_regions.remove(v)
                    self.parent[v] = u
                    processed.update([u, v])
            
            if not processed and best_neighbor:
                u = next(iter(best_neighbor)); v = best_neighbor[u]
                if v != -1:
                    w_uv = adj[u][v]
                    vol[u] += vol[v]
                    cut[u] = cut[u] + cut[v] - 2 * w_uv
                    for neighbor, w_vn in list(adj[v].items()):
                        if neighbor == u: continue
                        new_w = adj[u].get(neighbor, 0.0) + w_vn
                        adj[u][neighbor] = adj[neighbor][u] = new_w
                        if v in adj[neighbor]: del adj[neighbor][v]
                    adj[v] = {}; adj[u].pop(v, None)
                    active_regions.remove(v); self.parent[v] = u

        return self._generate_final_labels(H, W, n_pixels)

    def _generate_shifts(self):
        shifts = []
        for dy in range(-self.max_radius, self.max_radius + 1):
            for dx in range(-self.max_radius, self.max_radius + 1):
                if dy == 0 and dx == 0: continue
                if math.sqrt(dy**2 + dx**2) <= self.max_radius:
                    shifts.append((dy, dx))
        return shifts

    def _generate_final_labels(self, H, W, n_pixels):
        labels = np.zeros(n_pixels, dtype=np.int32)
        for i in range(n_pixels):
            r = i
            while self.parent[r] != r: r = self.parent[r]
            labels[i] = r
        unique = np.unique(labels)
        mapping = {r: i for i, r in enumerate(unique)}
        return np.array([mapping[r] for r in labels], dtype=np.int32).reshape(H, W)
