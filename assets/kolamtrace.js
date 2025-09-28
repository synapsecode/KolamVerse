class KolamAnimator {
    constructor(cfg) {
        this.fileInput = document.getElementById(cfg.fileInputId);
        this.previewImg = document.getElementById(cfg.previewImgId);
        this.placeholder = document.getElementById(cfg.placeholderId);
        this.animPlaceholder = document.getElementById(cfg.animPlaceholderId)
        this.shimmerLoader = document.getElementById(cfg.shimmerLoaderId)
        this.animateBtn = document.getElementById(cfg.animateBtnId);
        this.canvas = document.getElementById(cfg.canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.backBtn = document.getElementById(cfg.backBtnId);
        this.forwardBtn = document.getElementById(cfg.forwardBtnId);
        this.slider = document.getElementById(cfg.sliderId);
        this.frames = []; this.currentFrame = 0; this.liveInterval = null; this.navImg = new Image(); this.FRAMESKIP = 20; this.csvFile = null;
        this.streamImg = null; // holds MJPEG image element
        this.streamStartedAt = 0;
        this._bindEvents();
    }
    _bindEvents() {
        this.fileInput.addEventListener('change', () => this._previewSelected());
        this.animateBtn.addEventListener('click', () => this._handleUpload());
        this.backBtn.addEventListener('click', () => this._updateFrame(this.currentFrame - this.FRAMESKIP));
        this.forwardBtn.addEventListener('click', () => this._updateFrame(this.currentFrame + this.FRAMESKIP));
        this.slider.addEventListener('input', () => this._updateFrame(parseInt(this.slider.value)));
    }
    _previewSelected() {
        const file = this.fileInput.files[0];
        if (file) {
            this.previewImg.src = URL.createObjectURL(file);
            this.previewImg.classList.remove('hidden');
            this.placeholder.classList.add('hidden');
            this.animateBtn.disabled = false;
        }
    }
    async _handleUpload() {
        this.shimmerLoader.style.display = "flex";
        const file = this.fileInput.files[0];
        if (!file) return alert('Select a file.');
        this.animPlaceholder.style.display = "none";
        this._resetState();
        this.previewImg.src = URL.createObjectURL(file);
        getKolamDescription(file).catch(() => { });
        try {
            const fd = new FormData(); fd.append('file', file);
            const res = await fetch('/upload_kolam', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.error) return alert('Error: ' + data.error);
            this.csvFile = data.csv_file;
            this.shimmerLoader.style.display = "none";
            this._startLiveMJPEG(this.csvFile);
            this._pollSnapshots();
        } catch (err) { console.error('Upload failed:', err); }
    }
    _resetState() {
        this.frames = []; this.currentFrame = 0; this.backBtn.disabled = true; this.forwardBtn.disabled = true; this.animateBtn.disabled = true; this.slider.value = 0; this.slider.max = 0; if (this.liveInterval) clearInterval(this.liveInterval); this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    _startLiveMJPEG(csv) {
        if (this.liveInterval) clearInterval(this.liveInterval);
        this.streamImg = new Image();
        this.streamImg.crossOrigin = 'anonymous';
        this.streamStartedAt = performance.now();
        this.streamImg.onload = () => {
            // Start drawing once first frame decoded
            if (this.liveInterval) clearInterval(this.liveInterval);
            this.liveInterval = setInterval(() => {
                try { this.ctx.drawImage(this.streamImg, 0, 0, this.canvas.width, this.canvas.height); }
                catch (e) { /* ignore until frame available */ }
            }, 60); // ~16 fps is enough; MJPEG frames may be slower
        };
        this.streamImg.onerror = () => {
            console.warn('MJPEG stream failed to load; relying on snapshots only.');
        };
        this.streamImg.src = `/animate_kolam?csv_file=${encodeURIComponent(csv)}`;
    }
    async _pollSnapshots() {
        try {
            const snapRes = await fetch('/kolam_snapshots', { cache: 'no-store' });
            if (snapRes.status === 202) {
                // Animation still generating; poll faster initially
                setTimeout(() => this._pollSnapshots(), 600);
                return;
            }
            if (!snapRes.ok) { setTimeout(() => this._pollSnapshots(), 1500); return; }
            const snapData = await snapRes.json();
            if (!Array.isArray(snapData.frames) || snapData.frames.length === 0) {
                setTimeout(() => this._pollSnapshots(), 1200);
                return;
            }
            this.frames = snapData.frames;
            this.currentFrame = this.frames.length - 1;
            if (this.liveInterval) {
                clearInterval(this.liveInterval);
                this.liveInterval = null;
            }
            this._updateFrame(this.currentFrame);
            this.slider.max = this.frames.length - 1; this.slider.value = this.currentFrame;
            // Re-enable animate so user can re-run with another file quickly
            this.animateBtn.disabled = false;
        } catch (e) {
            setTimeout(() => this._pollSnapshots(), 1500);
        }
    }
    _updateFrame(idx) {
        if (this.frames.length === 0) return;
        this.currentFrame = Math.max(0, Math.min(idx, this.frames.length - 1));
        this.navImg.onload = () => this.ctx.drawImage(this.navImg, 0, 0, this.canvas.width, this.canvas.height);
        this.navImg.src = 'data:image/jpeg;base64,' + this.frames[this.currentFrame];
        this.slider.value = this.currentFrame;
        this.backBtn.disabled = this.currentFrame === 0;
        this.forwardBtn.disabled = this.currentFrame >= this.frames.length - 1;
    }
}

new KolamAnimator({
    fileInputId: 'kolam-upload',
    previewImgId: 'preview-img',
    placeholderId: 'preview-placeholder',
    animPlaceholderId: 'anim-placeholder',
    animateBtnId: 'animate-btn',
    canvasId: 'animationCanvas',
    backBtnId: 'back20',
    forwardBtnId: 'forward20',
    sliderId: 'frameSlider',
    shimmerLoaderId: 'shimmer-loader'
});

async function getKolamDescription(file) {
    try {
        const descEl = document.getElementById('kolamDescription');
        const section = document.getElementById('descriptionSection');
        section.classList.add('show'); section.style.display = 'block'; section.style.marginBottom = '60px'
        descEl.textContent = 'Analyzing kolam pattern...'; descEl.className = 'loading-description';
        const fd = new FormData(); fd.append('file', file);
        const baseRes = await fetch('/describe_kolam', { method: 'POST', body: fd });
        const baseJson = await baseRes.json();
        descEl.className = '';
        if (baseJson.success && baseJson.description) {
            descEl.textContent = baseJson.description;
        } else { descEl.textContent = 'Error: ' + (baseJson.error || 'Unable to analyze this kolam image.'); return; }
        const mode = getSelectedNarrationMode();
        const narrFd = new FormData(); narrFd.append('file', file);
        const params = new URLSearchParams({ include_image: 'false', compress: 'true', mode });
        const narrRes = await fetch('/describe_kolam_ai?' + params.toString(), { method: 'POST', body: narrFd });
        const narrJson = await narrRes.json();
        const sourceEl = document.getElementById('narrationSource');
        if (narrJson.narration) {
            if (narrJson.narration.trim() !== baseJson.description.trim()) {
                descEl.textContent += '\n\nNarrated Drawing Flow:\n' + narrJson.narration;
            }
            sourceEl.hidden = false; sourceEl.textContent = (narrJson.source || (narrJson.ai_used ? 'AI' : 'Offline')).toUpperCase();
        }
    } catch (err) {
        const descEl = document.getElementById('kolamDescription');
        descEl.className = '';
        descEl.textContent = 'Error: Failed to connect to server. Please make sure the server is running.';
    }
}
function getSelectedNarrationMode() { const sel = document.querySelector('input[name="nmode"]:checked'); return sel ? sel.value : 'offline'; }
