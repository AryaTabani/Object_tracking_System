import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ========== Utility: HOG feature extraction for lighting robustness ==========
def compute_hog(gray):
    # Use OpenCV HOGDescriptor on a patch
    winSize = (gray.shape[1]//8*8, gray.shape[0]//8*8)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    gray_resized = cv2.resize(gray, winSize)
    features = hog.compute(gray_resized)
    # reshape back to 2D approx (we can just return vector, but for matching we use raw intensity)
    # For simplicity, just return normalized gray + HOG mean
    return (gray_resized.astype(np.float32)/255.0 + features.mean()).astype(np.float32)

# ========== Core FFT-based correlation ==========
def fft_match(search, template, use_cuda=False):
    search_f = search.astype(np.float32)
    template_f = template.astype(np.float32)
    search_f -= np.mean(search_f)
    template_f -= np.mean(template_f)

    if use_cuda:
        try:
            # upload to GPU
            g_search = cv2.cuda_GpuMat()
            g_template = cv2.cuda_GpuMat()
            g_search.upload(search_f)
            g_template.upload(template_f)
            res = cv2.cuda.matchTemplate(g_search, g_template, cv2.TM_CCOEFF_NORMED)
            r = res.download()
            _, max_val, _, max_loc = cv2.minMaxLoc(r)
            return max_val, max_loc
        except:
            pass  # fallback to CPU

    # CPU FFT implementation
    sh, sw = search_f.shape
    th, tw = template_f.shape
    H, W = cv2.getOptimalDFTSize(sh), cv2.getOptimalDFTSize(sw)
    pad_search = np.zeros((H, W), np.float32)
    pad_template = np.zeros((H, W), np.float32)
    pad_search[:sh,:sw] = search_f
    pad_template[:th,:tw] = template_f

    F_search = cv2.dft(pad_search, flags=cv2.DFT_COMPLEX_OUTPUT)
    F_template = cv2.dft(pad_template, flags=cv2.DFT_COMPLEX_OUTPUT)
    F_template_conj = F_template.copy()
    F_template_conj[:,:,1] *= -1
    R = cv2.mulSpectrums(F_search, F_template_conj, 0)
    mag = np.sqrt(R[:,:,0]**2 + R[:,:,1]**2)
    mag[mag == 0] = 1e-9
    R[:,:,0] /= mag
    R[:,:,1] /= mag
    r = cv2.idft(R, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    r = r[:sh-th+1, :sw-tw+1]
    _, max_val, _, max_loc = cv2.minMaxLoc(r)
    return max_val, max_loc

# ========== Kalman + Tracker class ==========
class FFTTracker:
    def __init__(self, frame, init_bbox):
        x, y, w, h = init_bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template_gray = gray[y:y+h, x:x+w]
        self.bbox = (x, y, w, h)
        self.kalman = self._init_kalman(x + w/2, y + h/2)
        # check for CUDA availability
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def _init_kalman(self, cx, cy):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]], np.float32)
        kf.measurementMatrix = np.eye(2,4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32)*0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*0.5
        kf.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
        return kf

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox
        pred = self.kalman.predict()
        cx, cy = int(pred[0]), int(pred[1])

        search_radius = int(max(w, h) * 1.5)
        sx, sy = max(0, cx - search_radius), max(0, cy - search_radius)
        ex, ey = min(frame.shape[1], cx + search_radius), min(frame.shape[0], cy + search_radius)
        search_window = gray[sy:ey, sx:ex]

        # Precompute HOG for template once
        template_feat = compute_hog(self.template_gray)

        scales = [0.9, 1.0, 1.1]
        results = []

        def match_scale(scale):
            new_w, new_h = int(w*scale), int(h*scale)
            if new_w < 5 or new_h < 5:
                return (-1, (0,0), scale)
            tmpl_resized = cv2.resize(self.template_gray, (new_w, new_h))
            # Combine HOG info
            tmpl_feat = compute_hog(tmpl_resized)
            if search_window.shape[0] < new_h or search_window.shape[1] < new_w:
                return (-1, (0,0), scale)
            val, loc = fft_match(search_window, tmpl_feat, self.use_cuda)
            return (val, loc, scale)

        # Run in parallel for scales
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(match_scale, s) for s in scales]
            for f in futures:
                results.append(f.result())

        # Choose best
        best_val, best_loc, best_scale = max(results, key=lambda r: r[0])

        CONF_THRESH = 0.4
        if best_val < CONF_THRESH:
            # Occluded
            return (cx - w//2, cy - h//2, w, h)
        else:
            bx, by = best_loc
            nx = sx + bx
            ny = sy + by
            nw, nh = int(w*best_scale), int(h*best_scale)

            self.kalman.correct(np.array([[np.float32(nx+nw/2)], [np.float32(ny+nh/2)]]))

            # Update template slowly
            alpha = 0.1
            patch = gray[ny:ny+nh, nx:nx+nw]
            if patch.shape == self.template_gray.shape:
                self.template_gray = cv2.addWeighted(self.template_gray, 1-alpha, patch, alpha, 0)

            self.bbox = (nx, ny, nw, nh)
            return self.bbox

# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    cap = cv2.VideoCapture("videos/person1.mp4")  # webcam; replace with 'video.mp4' if needed
    ret, frame = cap.read()
    if not ret:
        print("Cannot open video source.")
        exit()

    init_bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")
    tracker = FFTTracker(frame, init_bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x, y, w, h = tracker.update(frame)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imshow("Advanced Tracking", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
