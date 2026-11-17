import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export const HeroVisual: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [, setModelLoaded] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e12);

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 1000);
    camera.position.set(0, 0, 20.5);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    container.appendChild(renderer.domElement);

    const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();

    // 2. Orbital Environment Reflection Map (Provides realistic orbital reflections for metals & solar arrays)
    const envCanvas = document.createElement('canvas');
    envCanvas.width = 256;
    envCanvas.height = 128;
    const envCtx = envCanvas.getContext('2d');
    if (envCtx) {
      // Deep space black
      envCtx.fillStyle = '#030712';
      envCtx.fillRect(0, 0, 256, 128);

      // Blue Earth atmospheric reflection in lower hemisphere (Earthshine)
      const earthGrad = envCtx.createRadialGradient(190, 100, 10, 190, 100, 75);
      earthGrad.addColorStop(0, '#38bdf8');
      earthGrad.addColorStop(0.4, '#0284c7');
      earthGrad.addColorStop(0.75, '#0f172a');
      earthGrad.addColorStop(1, '#030712');
      envCtx.fillStyle = earthGrad;
      envCtx.beginPath();
      envCtx.arc(190, 100, 75, 0, Math.PI * 2);
      envCtx.fill();

      // Warm solar reflection glint
      const sunGrad = envCtx.createRadialGradient(85, 30, 0, 85, 30, 24);
      sunGrad.addColorStop(0, '#ffffff');
      sunGrad.addColorStop(0.3, '#fef08a');
      sunGrad.addColorStop(0.65, '#f59e0b');
      sunGrad.addColorStop(1, 'rgba(3, 7, 18, 0)');
      envCtx.fillStyle = sunGrad;
      envCtx.beginPath();
      envCtx.arc(85, 30, 24, 0, Math.PI * 2);
      envCtx.fill();
    }

    const envTexture = new THREE.CanvasTexture(envCanvas);
    envTexture.mapping = THREE.EquirectangularReflectionMapping;
    const pmremGen = new THREE.PMREMGenerator(renderer);
    pmremGen.compileEquirectangularShader();
    const orbitalEnvMap = pmremGen.fromEquirectangular(envTexture).texture;
    scene.environment = orbitalEnvMap;

    // Coherent Cinematic Space Lighting
    const sunDirection = new THREE.Vector3(0.72, 0.58, 0.88).normalize();

    const sunLight = new THREE.DirectionalLight(0xfff6ec, 2.9);
    sunLight.position.copy(sunDirection).multiplyScalar(40);
    scene.add(sunLight);

    const ambientLight = new THREE.AmbientLight(0x0e1726, 0.65);
    scene.add(ambientLight);

    // Dedicated key and fill lights specifically for the satellite matching Earth's sun & earthshine
    const satPivot = new THREE.Group();
    scene.add(satPivot);

    // Direct warm solar key light (brightened for vibrant spacecraft illumination)
    const satSunLight = new THREE.DirectionalLight(0xfffaee, 4.2);
    satSunLight.position.set(15, 16, 14);
    satSunLight.target = satPivot;
    scene.add(satSunLight);

    // Direct wing specular illuminator (brightens photovoltaic solar wings)
    const satWingLight = new THREE.DirectionalLight(0x93c5fd, 2.5);
    satWingLight.position.set(8, 10, 12);
    satWingLight.target = satPivot;
    scene.add(satWingLight);

    // Earthshine: vibrant atmospheric blue reflection from the planet below
    const satEarthshineLight = new THREE.DirectionalLight(0x38bdf8, 2.2);
    satEarthshineLight.position.set(3, -5, 6);
    satEarthshineLight.target = satPivot;
    scene.add(satEarthshineLight);

    // Crisp rim light for solar array and antenna silhouettes
    const satRimLight = new THREE.DirectionalLight(0xffffff, 1.8);
    satRimLight.position.set(5, 7, 16);
    satRimLight.target = satPivot;
    scene.add(satRimLight);

    // 3. Sparse, Subtle Starfield
    const starCount = 220;
    const starGeometry = new THREE.BufferGeometry();
    const starPositions = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const radius = 90 + Math.random() * 80;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 1.7 - 0.85);

      starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = radius * Math.cos(phi);
      starPositions[i * 3 + 2] = -Math.abs(radius * Math.sin(phi) * Math.sin(theta)) - 10;

      const temp = Math.random();
      if (temp > 0.82) {
        starColors[i * 3] = 0.82;
        starColors[i * 3 + 1] = 0.9;
        starColors[i * 3 + 2] = 1.0;
      } else if (temp > 0.6) {
        starColors[i * 3] = 1.0;
        starColors[i * 3 + 1] = 0.95;
        starColors[i * 3 + 2] = 0.85;
      } else {
        starColors[i * 3] = 0.92;
        starColors[i * 3 + 1] = 0.94;
        starColors[i * 3 + 2] = 0.96;
      }
    }

    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

    const starMaterial = new THREE.PointsMaterial({
      size: 0.75,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      sizeAttenuation: true,
    });
    const starField = new THREE.Points(starGeometry, starMaterial);
    scene.add(starField);

    // 4. Earth Setup (Positioned in right bottom corner so only a quarter of Earth is visible)
    const earthRadius = 9.5;
    const earthCenterX = 8.5;
    const earthCenterY = -7.0;
    const earthGroup = new THREE.Group();
    earthGroup.position.set(earthCenterX, earthCenterY, 0);
    scene.add(earthGroup);

    // Texture Loading with Proper Responsibilities:
    // 1. Albedo: 4096x2048 4K Genuine Natural-Color Blue Marble Earth
    // 2. Normal: 2K Topography relief map
    // 3. Specular: Ocean water mask
    // 4. Night Lights: NASA Black Marble 2016 3km
    // 5. Clouds: Separate high-res transparent cloud layer
    const textureLoader = new THREE.TextureLoader();

    // 4K Genuine Natural-Color Earth Albedo for crisp coastlines and continental clarity
    const dayMap = textureLoader.load('/textures/earth-natural-4k.webp');
    dayMap.colorSpace = THREE.SRGBColorSpace;
    dayMap.anisotropy = maxAnisotropy;
    dayMap.minFilter = THREE.LinearMipmapLinearFilter;
    dayMap.magFilter = THREE.LinearFilter;
    dayMap.generateMipmaps = true;

    // High-Res 2048x1024 Topographic Normal Map
    const normalMap = textureLoader.load('/textures/earth-normal-2k.webp');
    normalMap.colorSpace = THREE.NoColorSpace;
    normalMap.anisotropy = maxAnisotropy;
    normalMap.minFilter = THREE.LinearMipmapLinearFilter;
    normalMap.generateMipmaps = true;

    // High-Res 2048x1024 Specular Map (Water = 1.0, Land = 0.0)
    const specularMap = textureLoader.load('/textures/earth-specular-2k.webp');
    specularMap.colorSpace = THREE.NoColorSpace;
    specularMap.anisotropy = maxAnisotropy;
    specularMap.minFilter = THREE.LinearMipmapLinearFilter;
    specularMap.generateMipmaps = true;

    // NASA Black Marble 2016 3km Night Lights
    const nightMap = textureLoader.load('/textures/earth-night-hires.webp');
    nightMap.colorSpace = THREE.SRGBColorSpace;
    nightMap.anisotropy = maxAnisotropy;
    nightMap.minFilter = THREE.LinearMipmapLinearFilter;
    nightMap.generateMipmaps = true;

    // 2048x1024 NASA Clouds Map
    const cloudsMap = textureLoader.load('/textures/earth-clouds.webp');
    cloudsMap.colorSpace = THREE.SRGBColorSpace;
    cloudsMap.anisotropy = maxAnisotropy;
    cloudsMap.minFilter = THREE.LinearMipmapLinearFilter;
    cloudsMap.magFilter = THREE.LinearFilter;
    cloudsMap.generateMipmaps = true;

    const earthGeometry = new THREE.SphereGeometry(earthRadius, 96, 96);

    // Clean Natural-Color Earth Surface Shader
    const earthShader = {
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vWorldPos;

        void main() {
          vUv = uv;
          vNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
          vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;

        uniform sampler2D uDay;
        uniform sampler2D uNormal;
        uniform sampler2D uSpecular;
        uniform sampler2D uNight;
        uniform vec3 uSunDir;
        uniform vec3 uCamPos;

        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vWorldPos;

        void main() {
          vec3 N = normalize(vNormal);
          vec3 V = normalize(uCamPos - vWorldPos);
          vec3 L = normalize(uSunDir);

          // 1. Natural-Color Earth Albedo (Un-distorted true color: natural blue ocean, green/brown continents)
          vec3 dayColor = texture2D(uDay, vUv).rgb;

          // 2. Topography / Mountain Relief via Normal Map
          vec3 mapNormal = texture2D(uNormal, vUv).xyz * 2.0 - 1.0;
          vec3 tangent = normalize(cross(vec3(0.0, 1.0, 0.0), N));
          if (length(tangent) < 0.001) tangent = vec3(1.0, 0.0, 0.0);
          vec3 bitangent = normalize(cross(N, tangent));
          vec3 perturbedN = normalize(tangent * mapNormal.x * 0.28 + bitangent * mapNormal.y * 0.28 + N * mapNormal.z);

          float ndl = dot(perturbedN, L);
          float daylight = smoothstep(-0.20, 0.60, ndl);
          float ambient = 0.24;

          // 3. Ocean Specular Sun Glint & Glancing Fresnel
          float oceanMask = texture2D(uSpecular, vUv).r;
          vec3 H = normalize(L + V);
          float NdotH = max(dot(perturbedN, H), 0.0);
          float oceanGlint = pow(NdotH, 36.0) * 0.45 * oceanMask * smoothstep(0.0, 0.25, ndl);
          vec3 specColor = vec3(0.85, 0.94, 1.0) * oceanGlint;

          // Grazing angle ocean Fresnel
          float cosTheta = max(dot(N, V), 0.0);
          float fresnel = pow(1.0 - cosTheta, 5.0) * 0.25 * oceanMask * max(ndl, 0.0);

          // 4. Combined Surface Color
          vec3 color = dayColor * (ambient + daylight * 1.05) + specColor + vec3(0.12, 0.38, 0.85) * fresnel;

          // 5. Night Side City Lights (twilight and shadow zone)
          float nightFactor = smoothstep(0.10, -0.18, ndl);
          vec3 nightLights = texture2D(uNight, vUv).rgb * vec3(1.0, 0.84, 0.48) * 1.85 * nightFactor;
          color += nightLights;

          // 6. Subtle Rayleigh Atmospheric Surface Scattering along illuminated limb
          float rim = 1.0 - max(dot(N, V), 0.0);
          float atmoRim = pow(rim, 4.2) * max(ndl * 0.7 + 0.3, 0.0);
          color += vec3(0.06, 0.34, 0.88) * atmoRim * 0.55;

          gl_FragColor = vec4(color, 1.0);
        }
      `,
    };

    const earthMaterial = new THREE.ShaderMaterial({
      vertexShader: earthShader.vertexShader,
      fragmentShader: earthShader.fragmentShader,
      uniforms: {
        uDay: { value: dayMap },
        uNormal: { value: normalMap },
        uSpecular: { value: specularMap },
        uNight: { value: nightMap },
        uSunDir: { value: sunDirection },
        uCamPos: { value: camera.position },
      },
    });

    const earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);
    // Controlled cinematic orientation: Positioned so India, Himalayas, and the Indian Ocean are prominently framed on the visible quarter
    earthMesh.rotation.set(-0.41, 2.93, -0.18);
    earthGroup.add(earthMesh);

    // 5. Separate Transparent Cloud Layer (Atmospheric depth, multi-scale formations, soft translucent falloff)
    const cloudsGeometry = new THREE.SphereGeometry(earthRadius + 0.08, 96, 96);
    const cloudsShader = {
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vWorldPos;

        void main() {
          vUv = uv;
          vNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
          vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;

        uniform sampler2D uClouds;
        uniform vec3 uSunDir;
        uniform vec3 uCamPos;

        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vWorldPos;

        // 3D Simplex Noise for procedural multi-scale atmospheric cloud formations
        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
        vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

        float snoise(vec3 v) {
          const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
          const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

          vec3 i  = floor(v + dot(v, C.yyy));
          vec3 x0 = v - i + dot(i, C.xxx);

          vec3 g = step(x0.yzx, x0.xyz);
          vec3 l = 1.0 - g;
          vec3 i1 = min(g.xyz, l.zxy);
          vec3 i2 = max(g.xyz, l.zxy);

          vec3 x1 = x0 - i1 + C.xxx;
          vec3 x2 = x0 - i2 + C.yyy;
          vec3 x3 = x0 - D.yyy;

          i = mod289(i);
          vec4 p = permute(permute(permute(
                     i.z + vec4(0.0, i1.z, i2.z, 1.0))
                   + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                   + i.x + vec4(0.0, i1.x, i2.x, 1.0));

          float n_ = 0.142857142857;
          vec3 ns = n_ * D.wyz - D.xzx;

          vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

          vec4 x_ = floor(j * ns.z);
          vec4 y_ = floor(j - 7.0 * x_);

          vec4 x = x_ * ns.x + ns.yyyy;
          vec4 y = y_ * ns.x + ns.yyyy;
          vec4 h = 1.0 - abs(x) - abs(y);

          vec4 b0 = vec4(x.xy, y.xy);
          vec4 b1 = vec4(x.zw, y.zw);

          vec4 s0 = floor(b0) * 2.0 + 1.0;
          vec4 s1 = floor(b1) * 2.0 + 1.0;
          vec4 sh = -step(h, vec4(0.0));

          vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
          vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

          vec3 p0 = vec3(a0.xy, h.x);
          vec3 p1 = vec3(a0.zw, h.y);
          vec3 p2 = vec3(a1.xy, h.z);
          vec3 p3 = vec3(a1.zw, h.w);

          vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
          p0 *= norm.x;
          p1 *= norm.y;
          p2 *= norm.z;
          p3 *= norm.w;

          vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
          m = m * m;
          return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
        }

        void main() {
          vec3 N = normalize(vNormal);
          vec3 L = normalize(uSunDir);
          vec3 V = normalize(uCamPos - vWorldPos);

          // Subtle domain-warp to soften texture boundaries into natural fluid fronts
          vec3 p = vWorldPos * 0.45;
          float warpX = snoise(p + vec3(0.0, 1.2, 2.4));
          float warpY = snoise(p + vec3(4.3, 0.0, 1.7));
          vec2 warpedUv = vUv + vec2(warpX, warpY) * 0.008;

          float rawDensity = texture2D(uClouds, warpedUv).r;

          // Multi-scale procedural detail: adds fine wisps and feathered banks to macro weather systems
          float fineNoise = snoise(vWorldPos * 1.8) * 0.12 + snoise(vWorldPos * 4.2) * 0.05;
          float combined = rawDensity + fineNoise;

          // Continuous smooth density falloff: clear regions remain transparent, edges are feather-soft
          float cloudDensity = smoothstep(0.20, 0.76, combined);
          cloudDensity = pow(cloudDensity, 1.35);

          // Directional solar lighting on clouds
          float ndl = dot(N, L);
          float sunLit = smoothstep(-0.12, 0.38, ndl);

          // Luminous natural white illuminated by sunlight, with airy sky-tinted ambient
          // (avoids dark gray patches or hard blobs)
          vec3 sunCloud = vec3(0.99, 1.0, 1.0);
          vec3 ambientCloud = vec3(0.88, 0.93, 0.98);
          vec3 cloudColor = mix(ambientCloud, sunCloud, 0.35 + sunLit * 0.65);

          // Forward Mie scattering on the atmospheric horizon limb
          float vDotL = dot(-V, L);
          float scatter = pow(max(vDotL, 0.0), 3.2) * 0.14 * sunLit;
          cloudColor += vec3(scatter);

          // Calibrated translucency (peak opacity ~0.30): leaves India, coastlines, and oceans fully distinct
          // Clouds naturally fade on the night terminator without forming gray film
          float nightFade = smoothstep(-0.15, 0.20, ndl);
          float alpha = cloudDensity * 0.30 * nightFade;

          gl_FragColor = vec4(cloudColor, alpha);
        }
      `,
    };

    const cloudsMaterial = new THREE.ShaderMaterial({
      vertexShader: cloudsShader.vertexShader,
      fragmentShader: cloudsShader.fragmentShader,
      uniforms: {
        uClouds: { value: cloudsMap },
        uSunDir: { value: sunDirection },
        uCamPos: { value: camera.position },
      },
      transparent: true,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });

    const cloudsMesh = new THREE.Mesh(cloudsGeometry, cloudsMaterial);
    cloudsMesh.rotation.set(earthMesh.rotation.x, earthMesh.rotation.y + 0.08, earthMesh.rotation.z);
    earthGroup.add(cloudsMesh);

    // 6. Restrained Atmospheric Rim (Thin, soft Rayleigh scattering haze along the planetary horizon)
    const atmosphereGeometry = new THREE.SphereGeometry(earthRadius + 0.10, 96, 96);
    const atmosphereShader = {
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;
        uniform vec3 sunDirView;
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
          vec3 viewDir = normalize(-vPosition);
          float rim = 1.0 - max(dot(vNormal, viewDir), 0.0);
          float ndl = dot(vNormal, sunDirView);

          // Sunlight factor on day side with smooth twilight transition
          float sunFactor = smoothstep(-0.25, 0.35, ndl);

          // Subtle blue Rayleigh rim with very slight twilight warmth along the limb
          vec3 daySky = vec3(0.24, 0.58, 0.96);
          vec3 twilightGlow = vec3(0.88, 0.50, 0.22);
          vec3 atmoColor = mix(twilightGlow, daySky, smoothstep(-0.15, 0.25, ndl));

          // Razor-thin, crisp, soft planetary rim
          float alpha = pow(rim, 5.2) * 0.52 * sunFactor;

          gl_FragColor = vec4(atmoColor, alpha);
        }
      `,
    };

    const atmosphereMaterial = new THREE.ShaderMaterial({
      vertexShader: atmosphereShader.vertexShader,
      fragmentShader: atmosphereShader.fragmentShader,
      uniforms: {
        sunDirView: { value: sunDirection },
      },
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
    });
    const atmosphereMesh = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
    earthGroup.add(atmosphereMesh);

    // 7. High-Resolution Earth Observation Satellite (Bright, vibrant photovoltaic wings, NASA Gold Kapton MLI, and optics)
    const satellitePosition = new THREE.Vector3(7.2, 4.2, 3.8);
    satPivot.position.copy(satellitePosition);

    const satellite = new THREE.Group();
    satellite.scale.setScalar(0.75);

    // Premium PBR Materials with high specular response and orbital environment reflections
    const goldKaptonMat = new THREE.MeshStandardMaterial({
      color: 0xd97706,
      metalness: 0.60,
      roughness: 0.25,
      envMapIntensity: 2.6,
    });
    const silverRadiatorMat = new THREE.MeshStandardMaterial({
      color: 0xcfd8dc,
      metalness: 0.70,
      roughness: 0.20,
      envMapIntensity: 2.2,
    });
    const darkCompositeMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      metalness: 0.45,
      roughness: 0.35,
      envMapIntensity: 1.5,
    });
    // Brighter, high-specular photovoltaic solar cells with luminous space blue
    const brightSolarMat = new THREE.MeshStandardMaterial({
      color: 0x2563eb,
      emissive: 0x0f2f64,
      metalness: 0.90,
      roughness: 0.08,
      envMapIntensity: 3.8,
    });
    const goldTrimMat = new THREE.MeshStandardMaterial({
      color: 0xfbbf24,
      metalness: 0.75,
      roughness: 0.18,
      envMapIntensity: 2.5,
    });
    const silverGridMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      metalness: 0.95,
      roughness: 0.08,
      envMapIntensity: 3.0,
    });
    const opticalGlassMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.90,
      roughness: 0.05,
      envMapIntensity: 3.5,
    });

    // Main Spacecraft Core (Clad in NASA Gold MLI Thermal Foil)
    const busGeo = new THREE.BoxGeometry(0.82, 0.60, 0.60);
    const busMesh = new THREE.Mesh(busGeo, goldKaptonMat);
    satellite.add(busMesh);

    // Cold-side Silver Equipment Radiator Louvers
    const radLeft = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.50, 0.02), silverRadiatorMat);
    radLeft.position.set(0, 0, 0.305);
    satellite.add(radLeft);

    const radRight = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.50, 0.02), silverRadiatorMat);
    radRight.position.set(0, 0, -0.305);
    satellite.add(radRight);

    // High-Resolution Articulated Dual Solar Array Wings (Extending laterally with bright photovoltaic blue cells)
    const createSolarWing = (direction: number) => {
      const wingGroup = new THREE.Group();

      // Dual-Axis Articulated Yoke
      const yoke = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.45), silverRadiatorMat);
      yoke.rotation.z = Math.PI / 2;
      yoke.position.set(direction * 0.52, 0, 0);
      wingGroup.add(yoke);

      // Solar Panel Frame with Gold Bevel Borders
      const wingFrameGeo = new THREE.BoxGeometry(1.65, 0.68, 0.035);
      const wingFrame = new THREE.Mesh(wingFrameGeo, goldTrimMat);
      wingFrame.position.set(direction * 1.50, 0, 0);
      wingGroup.add(wingFrame);

      // Bright Photovoltaic Cells (Front & Back)
      const cellsFront = new THREE.Mesh(new THREE.BoxGeometry(1.58, 0.62, 0.04), brightSolarMat);
      cellsFront.position.set(direction * 1.50, 0, 0);
      wingGroup.add(cellsFront);

      // Fine High-Res Grid Busbars (Silver electrical grid lines catching bright highlights)
      for (let i = 1; i <= 4; i++) {
        const vGrid = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.62, 0.045), silverGridMat);
        vGrid.position.set(direction * (0.85 + i * 0.28), 0, 0);
        wingGroup.add(vGrid);
      }
      for (let j = -1; j <= 1; j += 2) {
        const hGrid = new THREE.Mesh(new THREE.BoxGeometry(1.58, 0.015, 0.045), silverGridMat);
        hGrid.position.set(direction * 1.50, j * 0.18, 0);
        wingGroup.add(hGrid);
      }

      return wingGroup;
    };

    // Right and Left Solar Wings for realistic, balanced aerospace architecture
    const rightWing = createSolarWing(1);
    satellite.add(rightWing);

    const leftWing = createSolarWing(-1);
    satellite.add(leftWing);

    // Earth-Observation Multispectral Imaging Payload (Nadir-facing dual telescope aperture)
    const sensorPod = new THREE.Group();
    sensorPod.position.set(0, -0.38, 0);

    const barrelGeo = new THREE.CylinderGeometry(0.13, 0.16, 0.26, 24);
    const barrel = new THREE.Mesh(barrelGeo, darkCompositeMat);
    barrel.rotation.x = Math.PI / 2;
    sensorPod.add(barrel);

    const goldBezel = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.018, 12, 32), goldTrimMat);
    goldBezel.position.set(0, 0, 0.13);
    sensorPod.add(goldBezel);

    const lens = new THREE.Mesh(new THREE.CircleGeometry(0.11, 24), opticalGlassMat);
    lens.position.set(0, 0, 0.132);
    sensorPod.add(lens);

    satellite.add(sensorPod);

    // High-Gain Parabolic Communications Dish Antenna (Pointed toward Earth ground station)
    const dishGeo = new THREE.SphereGeometry(0.22, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.45);
    const dish = new THREE.Mesh(dishGeo, silverRadiatorMat);
    dish.position.set(-0.35, 0.24, -0.24);
    dish.rotation.x = -Math.PI / 3;
    dish.rotation.y = -Math.PI / 6;

    const dishFeed = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.18), goldTrimMat);
    dishFeed.position.set(-0.35, 0.24, -0.16);
    dishFeed.rotation.x = -Math.PI / 3;
    satellite.add(dish);
    satellite.add(dishFeed);

    // Star Tracker Navigation Sensor Pods
    const starTracker = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.06, 0.12, 16), darkCompositeMat);
    starTracker.position.set(0.28, 0.32, 0.15);
    starTracker.rotation.x = -Math.PI / 4;
    satellite.add(starTracker);

    // ACS Micro-Thruster Quads on corners
    const thrusterMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.7, roughness: 0.3 });
    [[-0.41, -0.28], [0.41, -0.28], [-0.41, 0.28], [0.41, 0.28]].forEach(([x, y]) => {
      const quad = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.03, 0.05, 8), thrusterMat);
      quad.position.set(x, y, 0);
      satellite.add(quad);
    });

    satPivot.add(satellite);
    setModelLoaded(true);

    // Natural 3/4 orbital angle catching the bright sun and camera view
    satPivot.rotation.set(-0.25, 0.45, -0.15);

    // 8. Animation Loop (Slow, cinematic, barely perceptible ambient orbital sway)
    let animationFrameId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      const elapsed = clock.getElapsedTime();

      if (!prefersReducedMotion) {
        earthMesh.rotation.y += 0.000025;
        cloudsMesh.rotation.y += 0.000032;

        const driftX = Math.cos(elapsed * 0.18) * 0.06;
        const driftY = Math.sin(elapsed * 0.22) * 0.04;

        satPivot.position.set(
          satellitePosition.x + driftX,
          satellitePosition.y + driftY,
          satellitePosition.z
        );

        // Subtle realistic orbital attitude sway
        satPivot.rotation.z = -0.15 + Math.sin(elapsed * 0.12) * 0.015;
        satPivot.rotation.x = -0.25 + Math.cos(elapsed * 0.15) * 0.012;
      }

      renderer.render(scene, camera);
      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    // 9. Responsive Resize Handler
    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;

      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();

      renderer.setSize(newWidth, newHeight);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

      if (newWidth < 768) {
        camera.position.z = 24;
        earthGroup.position.set(5.5, -9.0, 0);
        satPivot.position.set(4.0, 4.4, 2.0);
      } else {
        camera.position.z = 20.5;
        earthGroup.position.set(earthCenterX, earthCenterY, 0);
        satPivot.position.copy(satellitePosition);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    // 10. Cleanup and Disposal
    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);

      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }

      earthGeometry.dispose();
      earthMaterial.dispose();
      cloudsGeometry.dispose();
      cloudsMaterial.dispose();
      atmosphereGeometry.dispose();
      atmosphereMaterial.dispose();
      starGeometry.dispose();
      starMaterial.dispose();

      dayMap.dispose();
      normalMap.dispose();
      specularMap.dispose();
      nightMap.dispose();
      cloudsMap.dispose();

      envTexture.dispose();
      orbitalEnvMap.dispose();
      pmremGen.dispose();

      renderer.dispose();
    };
  }, []);

  return (
    <div className="absolute inset-0 w-full h-full overflow-hidden bg-[#0B0E12] pointer-events-none select-none z-0">
      <div ref={containerRef} className="absolute inset-0 w-full h-full" />
      <div className="absolute inset-0 bg-gradient-to-r from-[#0B0E12]/90 via-[#0B0E12]/40 to-transparent pointer-events-none" />
    </div>
  );
};
