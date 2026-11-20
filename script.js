import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ---------------------------------------------------------
// 1. Scene, Camera, Renderer & Post-Processing (Bloom)
// ---------------------------------------------------------
const container = document.getElementById('canvas-container');
const labelsLayer = document.getElementById('labels-layer');

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05070d, 0.018);

const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 250);
camera.position.set(0, 7.5, 25);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 8;
controls.maxDistance = 55;
controls.target.set(0, 0.5, 0);

const renderScene = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.85,
  0.45,
  0.22
);
const composer = new EffectComposer(renderer);
composer.addPass(renderScene);
composer.addPass(bloomPass);

// Subtle Lighting
scene.add(new THREE.AmbientLight(0x64748b, 1.4));
const dirLight = new THREE.DirectionalLight(0x93c5fd, 2.0);
dirLight.position.set(10, 20, 15);
scene.add(dirLight);

// Background Grid & Starfield
const gridHelper = new THREE.GridHelper(48, 32, 0x1e293b, 0x0f172a);
gridHelper.position.y = -5.5;
scene.add(gridHelper);

const starGeo = new THREE.BufferGeometry();
const starCount = 700;
const starPos = new Float32Array(starCount * 3);
for (let i = 0; i < starCount * 3; i++) {
  starPos[i] = (Math.random() - 0.5) * 120;
}
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
const starMat = new THREE.PointsMaterial({ color: 0x475569, size: 0.16 });
scene.add(new THREE.Points(starGeo, starMat));

// ---------------------------------------------------------
// 2. Simulation State & Deterministic Neural Math
// ---------------------------------------------------------
const state = {
  tokens: ['Transformers', 'preserve', 'old', 'memory', 'via', 'adapters'],
  selectedToken: 3,
  selectedHead: 'all',
  plasticityMode: 'adapter',
  trainSteps: 0,
  backboneDrift: 0,
  adapterStrength: 0.25
};

const HEAD_COLORS = [
  new THREE.Color(0x38bdf8),
  new THREE.Color(0xf59e0b),
  new THREE.Color(0xa855f7)
];

const LAYER_Y = {
  embed: -3.6,
  attn: -0.4,
  adapter: 2.3,
  mlp: 5.0
};

const networkGroup = new THREE.Group();
scene.add(networkGroup);

let interactiveMeshes = [];
let domLabels = [];
let pulseParticles = [];
let animatedRings = [];
let nodeRecords = [];

function seededWeight(a, b, c = 1) {
  const x = Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453;
  return x - Math.floor(x);
}

function computeAttentionMatrix(tokens) {
  const n = tokens.length;
  const heads = [[], [], []];
  for (let h = 0; h < 3; h++) {
    for (let i = 0; i < n; i++) {
      const row = [];
      let sum = 0;
      for (let j = 0; j < n; j++) {
        let score = 0.15;
        if (h === 0) {
          score += Math.exp(-Math.abs(i - j) * 0.85);
        } else if (h === 1) {
          const lenAffinity = (tokens[i].length * tokens[j].length) / 64;
          score += lenAffinity + seededWeight(i, j, h + 2) * 0.6;
        } else {
          if (j === 0 || j === n - 1 || i === j) score += 0.9;
          score += seededWeight(j, i, 9) * 0.5;
        }
        row.push(score);
        sum += score;
      }
      heads[h].push(row.map((v) => v / sum));
    }
  }
  return heads;
}

// ---------------------------------------------------------
// 3. Build 3D Graph (Nodes, Adapters, Attention Arcs, MLP)
// ---------------------------------------------------------
function clearNetwork() {
  while (networkGroup.children.length > 0) {
    networkGroup.remove(networkGroup.children[0]);
  }
  labelsLayer.innerHTML = '';
  interactiveMeshes = [];
  domLabels = [];
  pulseParticles = [];
  animatedRings = [];
  nodeRecords = [];
}

function buildNetwork() {
  clearNetwork();

  const tokens = state.tokens;
  const n = tokens.length;
  const spacing = Math.min(3.4, 21 / Math.max(n, 1));
  const startX = -((n - 1) * spacing) / 2;
  const attnMatrix = computeAttentionMatrix(tokens);

  const slabConfigs = [
    { y: LAYER_Y.embed, color: 0x1e293b },
    { y: LAYER_Y.attn, color: 0x1e3a5f },
    { y: LAYER_Y.adapter, color: 0x064e3b },
    { y: LAYER_Y.mlp, color: 0x311042 }
  ];

  slabConfigs.forEach((cfg) => {
    const slabGeo = new THREE.BoxGeometry(n * spacing + 3.2, 0.06, 5.4);
    const slabMat = new THREE.MeshBasicMaterial({
      color: cfg.color,
      transparent: true,
      opacity: 0.18
    });
    const slab = new THREE.Mesh(slabGeo, slabMat);
    slab.position.set(0, cfg.y, 0);
    networkGroup.add(slab);

    const edges = new THREE.EdgesGeometry(slabGeo);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: cfg.color, transparent: true, opacity: 0.45 })
    );
    line.position.copy(slab.position);
    networkGroup.add(line);
  });

  const incomingMag = new Array(n).fill(0);
  for (let h = 0; h < 3; h++) {
    if (state.selectedHead !== 'all' && Number(state.selectedHead) !== h) continue;
    for (let j = 0; j < n; j++) {
      incomingMag[j] += attnMatrix[h][state.selectedToken][j];
    }
  }
  const maxMag = Math.max(...incomingMag, 0.001);

  for (let i = 0; i < n; i++) {
    const x = startX + i * spacing;
    const isQuery = i === state.selectedToken;
    const normAct = incomingMag[i] / maxMag;

    const frozenColor = new THREE.Color(0x38bdf8).lerp(
      new THREE.Color(0xf43f5e),
      state.backboneDrift
    );

    // Embedding Node
    const embedRadius = 0.34 + normAct * 0.22;
    const embedMesh = new THREE.Mesh(
      new THREE.SphereGeometry(embedRadius, 24, 24),
      new THREE.MeshStandardMaterial({
        color: isQuery ? 0xffffff : frozenColor,
        emissive: isQuery ? 0x38bdf8 : frozenColor,
        emissiveIntensity: isQuery ? 1.4 : 0.35 + normAct * 0.6,
        roughness: 0.25,
        metalness: 0.6
      })
    );
    embedMesh.position.set(x, LAYER_Y.embed, 0);
    embedMesh.userData = { tokenIndex: i };
    networkGroup.add(embedMesh);
    interactiveMeshes.push(embedMesh);

    const labelEl = document.createElement('div');
    labelEl.className = 'node-label' + (isQuery ? ' active-token' : '');
    labelEl.textContent = `${tokens[i]} (${(normAct * 100).toFixed(0)}%)`;
    labelsLayer.appendChild(labelEl);
    domLabels.push({ el: labelEl, pos: new THREE.Vector3(x, LAYER_Y.embed - 0.55, 0) });

    // Attention Node
    const attnRadius = 0.38 + normAct * 0.35;
    const attnMesh = new THREE.Mesh(
      new THREE.OctahedronGeometry(attnRadius, 1),
      new THREE.MeshStandardMaterial({
        color: isQuery ? 0xf59e0b : frozenColor,
        emissive: isQuery ? 0xf59e0b : frozenColor,
        emissiveIntensity: isQuery ? 1.5 : 0.3 + normAct * 0.85,
        roughness: 0.2
      })
    );
    attnMesh.position.set(x, LAYER_Y.attn, 0);
    attnMesh.userData = { tokenIndex: i };
    networkGroup.add(attnMesh);
    interactiveMeshes.push(attnMesh);

    // Relational Adapter Ring
    const isAdapterMode = state.plasticityMode === 'adapter';
    const adapterScale = isAdapterMode ? 0.85 + state.adapterStrength * 0.55 : 0.55;
    const ringColor = isAdapterMode ? 0x10b981 : 0x475569;

    const torus = new THREE.Mesh(
      new THREE.TorusGeometry(0.55 * adapterScale, 0.075, 16, 36),
      new THREE.MeshStandardMaterial({
        color: ringColor,
        emissive: ringColor,
        emissiveIntensity: isAdapterMode ? 0.7 + normAct * 1.1 : 0.12
      })
    );
    torus.position.set(x, LAYER_Y.adapter, 0);
    torus.rotation.x = Math.PI / 2;
    networkGroup.add(torus);
    animatedRings.push({ mesh: torus, speed: (isAdapterMode ? 1.8 : 0.2) * (i % 2 === 0 ? 1 : -1) });

    const spine = new THREE.Mesh(
      new THREE.CylinderGeometry(0.11, 0.11, LAYER_Y.mlp - LAYER_Y.embed, 12),
      new THREE.MeshBasicMaterial({
        color: frozenColor,
        transparent: true,
        opacity: 0.28 + normAct * 0.25
      })
    );
    spine.position.set(x, (LAYER_Y.mlp + LAYER_Y.embed) / 2, 0);
    networkGroup.add(spine);

    // MLP Neurons
    for (let m = -1; m <= 1; m++) {
      const mlpAct = seededWeight(i, m + 2, state.selectedToken + 1) * normAct;
      const mlpRadius = 0.2 + mlpAct * 0.28;
      const mlpColor =
        state.backboneDrift > 0.35
          ? new THREE.Color(0xf43f5e)
          : new THREE.Color(0xa855f7).lerp(new THREE.Color(0x38bdf8), mlpAct);

      const mlpNode = new THREE.Mesh(
        new THREE.SphereGeometry(mlpRadius, 18, 18),
        new THREE.MeshStandardMaterial({
          color: mlpColor,
          emissive: mlpColor,
          emissiveIntensity: 0.25 + mlpAct * 1.4
        })
      );
      mlpNode.position.set(x + m * 0.42, LAYER_Y.mlp, m * 0.55);
      mlpNode.userData = { tokenIndex: i };
      networkGroup.add(mlpNode);
      interactiveMeshes.push(mlpNode);

      const pts = [new THREE.Vector3(x, LAYER_Y.adapter, 0), mlpNode.position.clone()];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lineMat = new THREE.LineBasicMaterial({
        color: isAdapterMode ? 0x10b981 : mlpColor,
        transparent: true,
        opacity: 0.18 + mlpAct * 0.65
      });
      networkGroup.add(new THREE.Line(lineGeo, lineMat));
    }

    nodeRecords.push({ embedMesh, attnMesh, normAct, isQuery });
  }

  // Multi-Head Attention Arcs & Signal Pulses
  const qIdx = state.selectedToken;
  const qX = startX + qIdx * spacing;

  for (let h = 0; h < 3; h++) {
    if (state.selectedHead !== 'all' && Number(state.selectedHead) !== h) continue;
    const zOffset = (h - 1) * 1.35;

    for (let kIdx = 0; kIdx < n; kIdx++) {
      const weight = attnMatrix[h][qIdx][kIdx];
      if (weight < 0.06) continue;

      const kX = startX + kIdx * spacing;
      const midX = (qX + kX) / 2;
      const arcDrop = -1.1 - Math.abs(kIdx - qIdx) * 0.35 - h * 0.25;

      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(qX, LAYER_Y.attn, 0),
        new THREE.Vector3(midX, LAYER_Y.attn + arcDrop, zOffset * 1.4),
        new THREE.Vector3(kX, LAYER_Y.attn, 0)
      );

      const tubeRadius = 0.018 + weight * 0.085;
      const tubeGeo = new THREE.TubeGeometry(curve, 28, tubeRadius, 8, false);
      const tubeMat = new THREE.MeshBasicMaterial({
        color: HEAD_COLORS[h],
        transparent: true,
        opacity: Math.min(0.95, 0.22 + weight * 1.35)
      });
      networkGroup.add(new THREE.Mesh(tubeGeo, tubeMat));

      const pulseCount = weight > 0.22 ? 2 : 1;
      for (let p = 0; p < pulseCount; p++) {
        const photon = new THREE.Mesh(
          new THREE.SphereGeometry(0.07 + weight * 0.08, 10, 10),
          new THREE.MeshBasicMaterial({ color: 0xffffff })
        );
        networkGroup.add(photon);
        pulseParticles.push({
          mesh: photon,
          curve,
          progress: p * 0.5 + seededWeight(qIdx, kIdx, h) * 0.3,
          speed: 0.35 + weight * 0.65
        });
      }
    }
  }
}

// ---------------------------------------------------------
// 4. Raycasting & UI Wiring
// ---------------------------------------------------------
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pointerDownPos = { x: 0, y: 0 };

renderer.domElement.addEventListener('pointerdown', (e) => {
  pointerDownPos = { x: e.clientX, y: e.clientY };
});

renderer.domElement.addEventListener('pointerup', (e) => {
  const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
  if (dist > 6) return;

  pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const hits = raycaster.intersectObjects(interactiveMeshes, false);
  if (hits.length > 0 && hits[0].object.userData.tokenIndex !== undefined) {
    selectToken(hits[0].object.userData.tokenIndex);
  }
});

function renderTokenDock() {
  const dock = document.getElementById('token-buttons');
  dock.innerHTML = '';
  state.tokens.forEach((tok, idx) => {
    const btn = document.createElement('button');
    btn.className = 'token-chip' + (idx === state.selectedToken ? ' selected' : '');
    btn.textContent = `[${idx}] ${tok}`;
    btn.onclick = () => selectToken(idx);
    dock.appendChild(btn);
  });
  document.getElementById('active-token-name').textContent =
    `Token [${state.selectedToken}]: "${state.tokens[state.selectedToken]}"`;
}

function selectToken(idx) {
  state.selectedToken = Math.max(0, Math.min(state.tokens.length - 1, idx));
  renderTokenDock();
  buildNetwork();
}

document.querySelectorAll('#head-selector button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#head-selector button').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.selectedHead = btn.dataset.head;
    const labels = {
      all: 'All Heads (H1–H3)',
      '0': 'H1 · Local Syntax',
      '1': 'H2 · Semantic Affinity',
      '2': 'H3 · Long-Range Causal'
    };
    document.getElementById('head-readout').textContent = labels[state.selectedHead];
    buildNetwork();
  });
});

document.querySelectorAll('#mode-selector button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#mode-selector button').forEach((b) => {
      b.classList.remove('active', 'emerald', 'rose');
    });
    state.plasticityMode = btn.dataset.mode;
    const modeReadout = document.getElementById('mode-readout');
    const statusText = document.getElementById('status-text');

    if (state.plasticityMode === 'adapter') {
      btn.classList.add('active', 'emerald');
      modeReadout.textContent = 'Adapter Update';
      modeReadout.style.color = 'var(--emerald)';
      statusText.innerHTML =
        '<strong>Weight Freeze + Adapter:</strong> Core weights W₀ stay frozen in cyan. Gradients route exclusively into emerald bottleneck rings (W₀ + BA), preserving prior tasks.';
      statusText.style.borderLeftColor = 'var(--emerald)';
    } else {
      btn.classList.add('active', 'rose');
      modeReadout.textContent = 'Full Fine-Tune';
      modeReadout.style.color = 'var(--rose)';
      statusText.innerHTML =
        '<strong>Unconstrained Fine-Tuning:</strong> Gradients modify the shared backbone directly. Click <em>Simulate New Task Gradient Step</em> to watch crimson drift overwrite old representations.';
      statusText.style.borderLeftColor = 'var(--rose)';
    }
    buildNetwork();
  });
});

document.getElementById('train-step-btn').addEventListener('click', () => {
  state.trainSteps += 1;
  document.getElementById('step-counter').textContent = `Step: ${state.trainSteps}`;
  const statusText = document.getElementById('status-text');

  if (state.plasticityMode === 'adapter') {
    state.adapterStrength = Math.min(1.0, state.adapterStrength + 0.18);
    statusText.innerHTML =
      `<strong>Step ${state.trainSteps} (Adapter Updated):</strong> Adapter capacity at <strong>${Math.round(state.adapterStrength * 100)}%</strong>. Backbone drift: <strong>${Math.round(state.backboneDrift * 100)}%</strong> (Zero forgetting).`;
  } else {
    state.backboneDrift = Math.min(1.0, state.backboneDrift + 0.22);
    statusText.innerHTML =
      `<strong>Step ${state.trainSteps} (Backbone Overwritten):</strong> Core parameter drift reached <strong>${Math.round(state.backboneDrift * 100)}%</strong> — earlier task pathways are degrading!`;
  }
  buildNetwork();
});

document.getElementById('reset-btn').addEventListener('click', () => {
  state.trainSteps = 0;
  state.backboneDrift = 0;
  state.adapterStrength = 0.25;
  document.getElementById('step-counter').textContent = 'Step: 0';
  buildNetwork();
});

document.getElementById('tokenize-btn').addEventListener('click', () => {
  const raw = document.getElementById('prompt-input').value.trim();
  if (!raw) return;
  const parsed = raw.split(/\s+/).slice(0, 8);
  state.tokens = parsed;
  state.selectedToken = Math.min(state.selectedToken, parsed.length - 1);
  renderTokenDock();
  buildNetwork();
});

// ---------------------------------------------------------
// 5. Animation Loop & Label Projection
// ---------------------------------------------------------
const clock = new THREE.Clock();

function updateLabels() {
  domLabels.forEach((item) => {
    const projected = item.pos.clone().project(camera);
    if (projected.z > 1) {
      item.el.style.opacity = '0';
      return;
    }
    const x = (projected.x * 0.5 + 0.5) * window.innerWidth;
    const y = (-projected.y * 0.5 + 0.5) * window.innerHeight;
    item.el.style.left = `${x}px`;
    item.el.style.top = `${y}px`;
    item.el.style.opacity = '1';
  });
}

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  const elapsed = clock.getElapsedTime();

  animatedRings.forEach((r) => {
    r.mesh.rotation.z += r.speed * dt;
  });

  nodeRecords.forEach((rec, idx) => {
    const pulse = 1 + Math.sin(elapsed * 3.5 + idx) * (0.04 + rec.normAct * 0.06);
    rec.attnMesh.scale.setScalar(pulse);
    rec.attnMesh.rotation.y += dt * (0.4 + rec.normAct);
  });

  pulseParticles.forEach((p) => {
    p.progress = (p.progress + dt * p.speed) % 1;
    const pt = p.curve.getPointAt(p.progress);
    p.mesh.position.copy(pt);
  });

  controls.update();
  updateLabels();
  composer.render();
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});

renderTokenDock();
buildNetwork();
animate();
