[**Access the full dataset and data viewer on Hugging Face here.**](https://huggingface.co/datasets/yatin-superintelligence/Audio-Video-Engineering-Agentic-Tasks-1M)

# Audio/Video Engineering Agentic Tasks (1M)

## Abstract

A highly specialized dataset featuring **1,029,459** in-context troubleshooting prompts and execution commands for the deepest levels of media production. Unlike standard datasets that simulate clean, theoretical instructions, this matrix captures the chaotic, highly-detailed, and conversational reality of professional audio engineers, composers, and video editors mid-session. It is engineered to train multimodal AI agents to operate in high-stress, technical environments where instructions are complex, multi-layered, and tightly coupled with time-based execution.

<img src="Audio Video Robot.jpg" alt="Audio Video Robot" width="100%"/>

## Infrastructure Architecture & Scale

This dataset bridges the gap between a deeply focused human operator and the precise, node-level actuations required to resolve their complex timeline issues.

* **Total Operations (Rows):** 1,029,459
* **Total Agent Task Tokens (Prompts only):** 156,015,488 (approx.)
* **Total Compute Expenditure (Generation):** 459,756,668 tokens (approx.)
* **Dimensionality:** 25 distinct professional archetypes strictly focused on audio engineering, music composition, and video post-production.
* **Storage Format:** Chunked Parquet (`batch_id` ordered, Zstandard compression)
* **Tokenization Basal Metric:** OpenAI `cl100k_base`
* **Operational Tone:** Conversational, diagnostic, multi-step — ranging from clinical and precise to high-pressure troubleshooting (exact average 127.75 words per instruction).

### Core Operational Objective: The "Mid-Session Crisis"

What makes this dataset entirely unique is its structural tone. The prompts are deliberately formulated as frantic, in-the-weeds diagnostic requests. A professional isn't just asking an agent to "add a compressor." They are explaining that the *"sustain is too long even when mutating hard enough,"* or that *"the nature texture just collapsed and everything is drifting,"* or that they are *"patching the VCA comp into the filter sweep path because the hardware riser module is missing."*

The defining challenge for the AI model is **Conversational Actuation**. The agent must parse through the context-heavy, domain-specific language — whether calm and technical or under pressure — to systematically deduce the exact sequence of GUI actuations, node connections, or harmonic adjustments required to resolve the immediate issue on the timeline.

### Representative Prompt Samples

To illustrate the density, high-stress conversational tone, and technical depth of this dataset, consider these four unedited rows mapping to different professional archetypes within their respective software environments:

#**1. Mixing Engineer (Bus Routing & Low-End Clash)**<br>
> *"The low end is a complete mess and I need you to open the sidechain compressor on the kick bus and set the threshold to -18 dB, attack at 2 ms, release at 80 ms. After that, route the bass guitar DI into the sidechain input so it ducks every time the kick hits — the two are completely canceling each other below 100 Hz right now and it's destroying the groove. Once the sidechain is locked in, pull up the multiband compressor on the master bus, isolate the 60–120 Hz band, and bring the ratio down to 2:1 so we stop over-compressing the sub. The mix should breathe again after that. If the stereo image still feels narrow, widen the mid channel on the aux reverb send by about 15% using the M/S matrix plugin."*

#**2. Sound Designer (Modular Synthesis & Filter Routing)**<br>
> *"The texture layer in patch 4 is way too static — I need you to route the output of the granular module directly into the input of the multimode filter, then set the filter to bandpass mode with a cutoff at 800 Hz and resonance at 65%. After that, patch an LFO running at 0.3 Hz into the filter cutoff CV input with a modulation depth of 40% so the texture breathes. The envelope follower from the drum bus should also be feeding into the VCA at the end of the chain so the whole texture pulses with the kick. Right now it's sitting flat on top of the mix and contributing nothing dynamically — once the modulation is active, automate the dry/wet on the granular module from 30% up to 80% over the 8-bar build."*

#**3. Video Editor (Multicam Sync & Pacing Fix)**<br>
> *"The multicam edit is completely out of sync from the 2:14 mark onwards — the angle 2 close-up is drifting about 3 frames behind the master audio track and it's getting worse as the sequence progresses. Go into the multicam clip in the timeline, open the angle editor, and manually offset angle 2 by +3 frames relative to the sync point at the clap on bar 1. After that I need you to recut the chorus section starting at 3:08 — the current cut is staying on the wide shot for 12 seconds which kills the energy. Switch to the close-up on the vocalist every 2 bars through the chorus and end each phrase on the beat with a hard cut, not a dissolve. The dissolves are making it feel slow."*

#**4. Colorist / Color Grader (Shot Match & LUT Override)**<br>
> *"The exterior shots in act 2 are completely mismatched with the interior coverage — the skies are blowing out at the highlights and the skin tones on the lead are going orange compared to the interior scenes which are much cooler. Pull up the parade scope on EXT_047 and bring the highlights down to 85 on the luma channel using the lift/gamma/gain wheels. Then go into the color warper and pull the orange-yellow vector inward toward neutral by about 25% to fix the skin tone shift. Once that shot is graded, use it as the reference and apply a shot match to the remaining 11 exterior clips in that sequence. After the match, check that none of them are clipping above 90 on the parade — if they are, add a secondary qualifier on the sky region and independently pull those highlights down."*

## Parquet Schema & Encoding

The dataset schema is normalized for high-throughput data loading in multi-GPU training environments:
| Schema Node | Data Type | Implementation Details |
| :--- | :--- | :--- |
| `batch_id` | Int64 | Chronological generation marker. Acts as the primary deterministic seed for combinatorial permutation. |
| `index` | Int64 | The matrix reference index corresponding to the specific tool combination chosen from the taxonomy. |
| `professional` | String | The exact occupational archetype (e.g.,`Orchestral Composer / Arranger`, `Foley Artist`). Determines the lexical bounds. |
| `group` | String | The overarching macro-category enabling stratified sampling. |
| `user_prompt` | String | The payload: The unformatted, raw task sequence requested of the multimodal agent. |

## Algorithm of Combinatorial Generation

To algorithmically ensure zero task duplication and eliminate monotonic semantic drift across 1.03 million rows, this dataset was generated via a multi-stage deterministic combinatorial engine:

**1. High-Fidelity Domain Taxonomies**<br>
The 25 archetypes are mapped against an expansive, highly specific software taxonomy centered on sound synthesis, timeline editing, color grading, and mixing architecture. Generic terms were purged and replaced with high-fidelity, node-level terminology (e.g., `dynamic_spectrum_analyzer`, `multicam_sync_bin`, `binaural_panner`).

**2. Pseudo-Random Intersection (The Tool Picker)**<br>
For each agent task, the deterministic engine selects precise tool combinations tailored to each professional. Over 1,000 parallel workers concurrently generated this data stream, dynamically choosing tools while bound to the `batch_id` seed to guarantee reproducibility and prevent combinatorial overlap.

By strategically sampling 1,029,459 times from the expansive pool of possible taxonomic intersections, the dataset achieves high structural entropy while remaining computationally reproducible.

**3. Hash-Based Behavioral Injection**<br>
To shatter the monotonic voice often found in synthetic datasets, the task engine utilizes a multi-parameter behavioral array. The behavior injected into the system prompt is strictly tied to a mathematical hash of the batch index. This ensures the prompts range dynamically from clinical problem-solving to complete exasperation (e.g. *"I CAN'T BELIEVE THE BASS IS STILL OVERPOWERING THE MIX"*).

## The 2 Multimodal Environments

Unlike broader generalized datasets, this collection focuses exclusively on the two most demanding time-based multithreaded environments.

### Agent Environment: Digital Audio Workstations (DAW) & Sound Engineering

* **Target Roles:** Music Producer, Mixing Engineer, Mastering Engineer, Recording Engineer, Sound Designer, Live Sound Engineer (In-the-Box), Audio Restoration Engineer, Game Audio Designer, Foley Artist, Broadcast Audio Engineer.
* **Target Roles (Compositional):** Orchestral Composer / Arranger, Film & TV Composer, Jingle & Commercial Composer, Electronic Artist / DJ Producer, Singer-Songwriter, Vocalist / Performer, Beat Maker.
* **Agent Operations:** The DAW represents an entirely invisible ecosystem governed by psychoacoustics, frequency spectra, and shifting dynamic ranges, forcing the agent to translate dense, abstract visual telemetry into perfect sonic outcomes. The agent must understand the physics of overlapping soundwaves, the harmonic characteristics of engineered distortion, and the structural depth of three-dimensional mixing spaces. A competent agent is expected to parse the emotional intent behind a sprawling, multi-layered musical arrangement, diagnosing structural sonic clashes, orchestrating vast routing architectures, and executing surgical interventions that elevate the artistic soul of the composition while adhering to unforgiving technical delivery standards.

### Agent Environment: Nonlinear Editing (NLE) & Cinematic Post-Production

* **Target Roles:** Video Editor, Colorist / Color Grader, Motion Graphics Artist, Documentary Filmmaker, Podcast Producer, YouTube / Content Creator, Music Educator / Trainer.
* **Agent Operations:** In these timeline-driven workspaces, reality is segmented into millions of sequential frames representing layered visual and auditory data streams. A multimodal agent must develop strong temporal awareness, grasping the psychological impact of pacing, editorial rhythm, and the emotional weight of manipulated color gamuts. The human user expects the agent to organize vast, unstructured media, sync decoupled data streams based on contextual observation, and execute technically precise aesthetic transformations that carry the narrative continuity of a cinematic vision from the first cut to the final master.

## Application Context

The **Audio/Video Engineering Agentic Tasks** dataset serves as core infrastructure for advanced AI models specializing in media generation, timeline orchestration, and creative post-production. Beyond powering standard visual action agents, this dataset isolates the specific difficulty of time-domain workflows—where changes on frame 1 ripple computationally to minute 40.

A critical application of this dataset is infrastructure generation: developers can use a frontier AI model to instantly spin up a custom, isolated software environment tailored to a specific professional archetype (e.g., a multi-track virtual mixing console or a simulated 8K video timeline). Because scaffolding this customized training infrastructure is achievable today, AI agents can be deployed into these bespoke environments to execute the tasks within this dataset, learning iteratively from their own interaction logs and structural feedback loops with a human of their choice.

## Architect & Developer

This dataset, pipeline architecture, and underlying generation framework were engineered and bounded by Yatin Taneja.

As an AI System Engineer, Superintelligence Researcher, Dubstep artist, Rapper and poet, the motivation behind this dataset is highly personal. Professional DAW and NLE environments are some of the most technically dense software ecosystems in existence — and until now, AI agents have had almost no structured training data to operate inside them. This dataset exists to close that gap. The 25 professionals in this collection represent the full chain of audio and visual production, from the composer writing stems to the colorist locking the final grade, and every instruction in this dataset reflects the real vocabulary they use — not a simplified version of it. The long-term goal is an agent that can sit inside a session as a capable collaborator: one that knows the difference between a technical and qualitative problem, and acts on that distinction without being explicitly told. I wish all the best to every AI developer and researcher out there!

### Weblinks

- **[IM Superintelligence](https://www.imsuperintelligence.ai):** Visit my central knowledge hub hosting other massive open datasets and over 2,000 articles exploring Superintelligence, cognitive architectures, quantum computing, distributed networks, algorithmic optimization, and the future of the global education sector, all authored through a custom 8-step multi-model agentic infrastructure I engineered.
- **[Yatin Taneja | Professional Portfolio](https://www.yatintaneja.in):** View my professional portfolio for a comprehensive overview of my skills, industry experience, and software prototypes as part of my ongoing engineering work in full-stack AI agents and applications.
- **[LinkedIn](https://www.linkedin.com/in/yatintaneja-pro/):** Connect on LinkedIn to collaborate on advanced autonomous systems, enterprise AI implementations, or to follow my ongoing research.


## License & Usage

This dataset is released under the **MIT License**.
Designed for advanced research into multimodal media production, timeline orchestration, and high-fidelity DAW/NLE actuation. You are free to use this dataset for commercial, academic, and personal model training focused on resolving complex audio-visual troubleshooting and execution tasks, provided the original license and copyright notice are preserved.