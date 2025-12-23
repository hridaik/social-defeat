import numpy as np
from PIL import Image, ImageDraw, ImageFont

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

# decay posterior probabilities towards a baseline - in this case  uniform
def decay_qs(ps, fr): # resembles leaky integration 
    qs = np.asarray(ps, dtype=float)
    n_elements = len(qs)
    uniform = np.ones(n_elements)/n_elements
    decayed = (1-fr)*qs + fr*uniform
    decayed /= decayed.sum()
    return decayed

# Build arena mask + mappings
def make_two_rooms_with_corridor(rows=3, left_cols=2, corridor_cols=3, right_cols=2,
                                 corridor_rows=(1),
                                 prefer_total_cols=None):
    
    default_total = left_cols + corridor_cols + right_cols
    if prefer_total_cols is None:
        total_cols = default_total
    else:
        total_cols = prefer_total_cols

    mask = np.zeros((rows, total_cols), dtype=bool)


    left_start = 0
    left_end = left_start + left_cols
    mask[:, left_start:left_end] = True

    corridor_start = left_end
    corridor_end = corridor_start + corridor_cols
    if corridor_end > total_cols:
        raise ValueError("Corridor doesn't fit in prefer_total_cols; increase total width or reduce corridor width.")
    for r in corridor_rows:
        mask[r, corridor_start:corridor_end] = True


    right_end = total_cols
    right_start = right_end - right_cols
    mask[:, right_start:right_end] = True

    return mask, {"left": (slice(None), slice(left_start, left_end)),
                  "corridor": (slice(None), slice(corridor_start, corridor_end)),
                  "right": (slice(None), slice(right_start, right_end))}





class grid:
    def __init__(self, mask: np.ndarray):

        self.mask = mask
        self.rows, self.cols = mask.shape
        self.valid_rc = [(r, c) for r in range(self.rows) 
                         for c in range(self.cols) 
                         if mask[r, c]]
        
        self.n_states = len(self.valid_rc)

        # maps: rc -> state id (0..n_states-1), and state->(r,c)
        self._rc_to_state = -np.ones((self.rows, self.cols), dtype=int) # -1 for blocked cell 
        self._state_to_rc = [None] * self.n_states

        for i, (r, c) in enumerate(self.valid_rc):
            self._rc_to_state[r, c] = i
            self._state_to_rc[i] = (r, c)


    def state_idx_to_rc(self, idx: int) -> tuple:
        return self._state_to_rc[int(idx)]

    def rc_to_state_idx(self, r: int, c: int) -> int:
        return int(self._rc_to_state[r, c])

    def is_valid(self, r: int, c: int) -> bool:
        return self.rc_to_idx(r, c) != -1
    
    def step_from_state(self, state_idx, action):
        r, c = self.state_idx_to_rc(state_idx)
        if action == 0:   # up
            nr, nc = r - 1, c
        elif action == 1: # down
            nr, nc = r + 1, c
        elif action == 2: # left
            nr, nc = r, c - 1
        elif action == 3: # right
            nr, nc = r, c + 1
        elif action == 4: # stay
            nr, nc = r, c
        else:
            raise ValueError("bad action")

        # If target is outside bounds or blocked, stay in place
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            return state_idx
        if not self.mask[nr, nc]:
            return state_idx
        
        return self.rc_to_state_idx(nr, nc)
    
    # compute manhattan distance using rc positions since no diagonal moves allowed
    def manhattan_states(self, s1, s2):
        r1, c1 = self.state_idx_to_rc(s1)
        r2, c2 = self.state_idx_to_rc(s2)
        return abs(r1 - r2) + abs(c1 - c2)
    
    # collect valid states in a column index
    def collect_states_in_column(self, col_idx):
        states = []
        for r in range(self.rows):
            if 0 <= col_idx < self.cols and self.mask[r, col_idx]:
                states.append(self.rc_to_state_idx(r, col_idx))
        return sorted(states)
    
    def fmt_topk_posterior(self, q_vec, k=3):
        q = np.asarray(q_vec, dtype=float)
        idx = np.argsort(q)[-k:][::-1]
        return [(int(i), float(q[i]), self.state_idx_to_rc(int(i))) for i in idx]

# Generative process (environment) class
class world_env():
    def __init__(self, arena, true_agent_pos, true_threat_pos, true_shelter_pos):
        self.arena = arena
        self.starting_pos = arena.rc_to_state_idx(true_agent_pos[0], true_agent_pos[1])
        self.agent_pos = self.starting_pos
        self.threat_pos = true_threat_pos
        self.threat_pos_list = [arena.rc_to_state_idx(pos[0], pos[1]) for pos in [(3, 13), (3, 14), (4, 13), (4, 14)]]
        self.shelter_pos = true_shelter_pos
        print(f'Initialized - Agent postion: {arena.state_idx_to_rc(self.starting_pos)}, Threat position: {arena.state_idx_to_rc(self.threat_pos)}')
        print()

    def step(self, action):
        new_state_idx = self.arena.step_from_state(self.agent_pos, action)
        self.agent_pos = new_state_idx
        agent_obs = self.agent_pos

        d_a_t = min([self.arena.manhattan_states(self.agent_pos, t_pos) for t_pos in self.threat_pos_list])
        threat_obs = max(10 - d_a_t, 0)

        d_a_s = 1000 # np.inf
        for s_pos in self.shelter_pos:
            d_a_si = self.arena.manhattan_states(self.agent_pos, s_pos)
            if (d_a_si < d_a_s):
                d_a_s = d_a_si
        shelter_obs = max(13 - d_a_s, 0)

        return agent_obs, threat_obs, shelter_obs
    
    def start(self):
        self.agent_pos = self.starting_pos
        agent_obs, threat_obs, shelter_obs = self.step(4) # stay
        return agent_obs, threat_obs, shelter_obs


# helper - visualization renderer
def render_grid_frame_arena(agent_state, shelter_state, visited_states, step,
                            threat_state=None, threat_posterior=None, cell_size=48,
                            threat_state_list=None, high_level_mode=None, current_scale=None, 
                            D_posterior=None, D_step=None, arena=None):
    """
    Draw arena with a colored status strip above it.
    The strip background color is GREEN when mode is safe, RED when danger.
    All strip text is white for readability.

    Args:
      - high_level_mode: optional string like "SAFE" or "DANGER" (preferred).
      - current_scale: optional tuple (threat_scale, shelter_scale).
      - high_posterior: optional vector [p_safe, p_danger] or scalar p_danger.
      - high_step: optional integer (high-level timestep) to show alongside low-level step.
    """

    # normalize shelter_state
    if isinstance(shelter_state, (list, tuple, np.ndarray, set)):
        shelter_set = set(int(x) for x in shelter_state)
    else:
        shelter_set = {int(shelter_state)}

    W = arena.cols * cell_size
    H = arena.rows * cell_size

    # --- draw grid ---
    img_grid = Image.new('RGB', (W, H), (255,255,255))
    draw = ImageDraw.Draw(img_grid)

    for r in range(arena.rows):
        for c in range(arena.cols):
            x0, y0 = c * cell_size, r * cell_size
            x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
            if not arena.mask[r, c]:
                fill = (50,50,50)
            else:
                idx = arena.rc_to_state_idx(r, c)
                if idx in visited_states and idx not in shelter_set and idx != agent_state and idx != threat_state:
                    fill = (220,220,220)
                else:
                    fill = (255,255,255)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(0,0,0))

    # heatmap overlay (if provided)
    if threat_posterior is not None:
        heat = np.zeros((arena.rows, arena.cols))
        for s_idx in range(len(threat_posterior)):
            r,c = arena.state_idx_to_rc(s_idx)
            heat[r,c] = float(threat_posterior[s_idx])
        for r in range(arena.rows):
            for c in range(arena.cols):
                if arena.mask[r, c] and heat[r,c] > 0:
                    alpha = min(0.9, heat[r,c]*2.5)
                    overlay = Image.new('RGBA', (cell_size, cell_size), (255,0,0,int(alpha*200)))
                    img_grid.paste(overlay, (c*cell_size, r*cell_size), overlay)

    # shelters
    for s_idx in shelter_set:
        r, c = arena.state_idx_to_rc(s_idx)
        draw.rectangle([c*cell_size, r*cell_size, (c+1)*cell_size-1, (r+1)*cell_size-1],
                       fill=(150,255,150), outline=(0,0,0))

    # threat
    if threat_state is not None:
        r, c = arena.state_idx_to_rc(int(threat_state))
        draw.rectangle([c*cell_size, r*cell_size, (c+1)*cell_size-1, (r+1)*cell_size-1],
                       fill=(255,150,150), outline=(0,0,0), width=2)
        try:
            draw.text((c*cell_size+cell_size//3, r*cell_size+cell_size//4), "T", fill=(0,0,0))
        except Exception:
            pass

    if threat_state_list is not None:
        threat_pos = [(3, 13), (3, 14), (4, 13), (4, 14)] # hardcoded for now, icba
        for (r, c) in threat_pos:
            draw.rectangle([c*cell_size, r*cell_size, (c+1)*cell_size-1, (r+1)*cell_size-1],
                        fill=(255,150,150), outline=(0,0,0), width=2)

    # agent
    if agent_state is not None:
        r, c = arena.state_idx_to_rc(int(agent_state))
        draw.rectangle([c*cell_size, r*cell_size, (c+1)*cell_size-1, (r+1)*cell_size-1],
                       fill=(30,30,200), outline=(0,0,0), width=2)
        try:
            draw.text((c*cell_size+cell_size//3, r*cell_size+cell_size//4), "A", fill=(255,255,255))
        except Exception:
            pass

    # Top status strip (single color: green for safe, red for danger)
    strip_h = max(46, cell_size // 2)
    final_H = H + strip_h
    final_img = Image.new('RGB', (W, final_H), (255,255,255))
    draw_final = ImageDraw.Draw(final_img)

    # font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    # Decide mode color:
    # Priority: use high_level_mode string if provided; else use high_posterior (p_danger threshold 0.5); default SAFE.
    mode_is_danger = False
    if high_level_mode is not None:
        try:
            mm = str(high_level_mode).lower()
            mode_is_danger = ('danger' in mm) or ('panic' in mm) or ('alert' in mm)
        except Exception:
            mode_is_danger = False
    elif D_posterior is not None:
        try:
            hp = np.array(D_posterior).ravel()
            p_d = float(hp[1]) if hp.size > 1 else float(hp[0])
            mode_is_danger = (p_d > 0.5)
        except Exception:
            mode_is_danger = False
    else:
        mode_is_danger = False

    strip_color = (180, 30, 30) if mode_is_danger else (30, 160, 50)  # red-ish or green-ish
    draw_final.rectangle([0, 0, W, strip_h], fill=strip_color)

    # White text for visibility
    text_color = (255,255,255)
    pad_x = 12
    y_text = (strip_h - 16) // 2

    # Left text: Step low (high)
    high_step = None
    if high_step is None:
        left_text = f"Step: {int(step)}"
    else:
        left_text = f"Step: {int(step)} ({int(high_step)})"
    draw_final.text((pad_x, y_text), left_text, fill=text_color, font=font)

    # # Mode text (slightly right)
    # mode_text = f"Mode: {str(high_level_mode) if high_level_mode is not None else ('DANGER' if mode_is_danger else 'SAFE')}"
    # draw_final.text((pad_x + 220, y_text), mode_text, fill=text_color, font=font)

    # # Scale text (center area)
    # if current_scale is None:
    #     scale_text = "Scale: N/A"
    # else:
    #     try:
    #         s0, s1 = current_scale
    #         scale_text = f"Scale: ({s0:.2f}, {s1:.2f})"
    #     except Exception:
    #         scale_text = f"Scale: {current_scale}"
    # draw_final.text((pad_x + 420, y_text), scale_text, fill=text_color, font=font)

    # # P(danger) numeric (right side)
    # if high_posterior is not None:
    #     try:
    #         hp = np.array(high_posterior).ravel()
    #         p_d = float(hp[1]) if hp.size > 1 else float(hp[0])
    #         pd_text = f"P(danger): {p_d:.2f}"
    #     except Exception:
    #         pd_text = "P(danger): N/A"
    # else:
    #     pd_text = "P(danger): N/A"
    # # right align near the right edge
    # try:
    #     tw, th = font.getsize(pd_text)
    # except Exception:
    #     tw, th = (len(pd_text)*7, 12)
    # draw_final.text((W - pad_x - tw, y_text), pd_text, fill=text_color, font=font)

    # Paste the grid under the strip
    final_img.paste(img_grid, (0, strip_h))

    return np.array(final_img)