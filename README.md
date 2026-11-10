# Reaching Lab

A window app for whole reaching-task experiment sessions filmed with one or two cameras (two GoPro HERO13 recommended): MediaPipe tracks the upper or lower limb, the 5 switches on the table and their lights are found in the video, every light becomes a trial synced with the experiment design, and the measures (RT, MT, ...) are written to CSV and Excel, the same method as the previous MATLAB system.

## Videos
The videos are not in git (too big); the 10 raw lab videos (IMG_0119-0123, IMG_7554-7558, ~300 MB each) are files of the GitHub release **lab-videos** of this (private) repo. To get them into `media/`:
```
gh release download lab-videos -R ofir-code/ai-camera-lab -D media
```
(or download them from the repo's Releases page). These are phone videos from before the GoPros: one camera each in the Lab App.

## Reaching Lab app (experiment sessions)
A window app for whole experiment sessions: double-click **`Lab App.command`** (Mac) or **`Lab App.bat`** (Windows), or run `venv/bin/python lab_app.py`.

**Per participant: 2 videos per day** (one per camera, each covering the whole session) for day 1, and 2 more for the re-test day. Everything else runs by itself.

1. **New participant…** - the ID, the design (tested arm, contralateral test yes/no, practice limb and side; the plan of each day is shown, e.g. Familiarization (2) → pre_test (3) → practice (10) → post_test (3) = 18 blocks; re-test day: re_test (3)) and the videos of day 1 / day 2 (they can be added later). *Settings* (usually left as they are): trials per sequence (6), sequences per block (5), light window (1 s), pause that starts a new block (10 s), switch order (found automatically), the board's measured square size (38 mm).
2. The app runs the day's **steps** one after the other; the bar on top of the day shows each step (✓ done, ! done with warnings, ✗ needs you, ⟳ running). Click a step to see what it found and to change it - after a change, the steps after it run again automatically ("Run this step again" re-runs a step from scratch).
   1. **Videos & sync** - the GoPro chapters, each camera's settings (notes if not as recommended below), and the offset between the cameras from their timecodes. A video without a timecode → that day is analysed with **camera A only (2D)** (or type the offset by hand). *Swap A / B* exchanges the cameras.
   2. **Calibration** - finds the ChArUco board in both videos (first 3 minutes, then minutes 3-7) and computes where the cameras are (3D, cm), with a check (the board's squares measured from the 3D result, reprojection error). Can be changed to: another time window, separate board clips, the calibration file of an earlier recording (cameras not moved), or no 3D. Board not found → each camera is analysed in 2D (warning).
   3. **Switches** - the 5 switches found from their lights (first 5 minutes, then minutes 5-10), numbered from the subject's right to left. If not all 5 are found the day stops here (✗): draw the missing box, *Flip numbering*, *Redraw* / *Remove* a box, or *Find again* from other minutes, then *Done - continue*.
   4. **Processing** - one pass over each video (MediaPipe pose, the switch lights, hand/foot positions), saved so it runs once per video. On a Mac about 1.5 x the video's length per camera (hardware video decoding + 3-4 MediaPipe workers). The video is cut into parts that run at the same time, one worker per 2 logical CPUs (`LAB_WORKERS=6` sets another number). On Windows the graphics card can decode the video with `LAB_HW_DECODE=1` (off by default: it converts the colours slightly differently, which can change the measured light levels - compare it once on a real session first; `LAB_HW_DECODE=0` turns it off on a Mac).
   5. **Blocks & trials** - *Trials*: like the old dataViewer: every trial with its outcome and marks, RT and MT, a *touse* tick box (the results are saved again by themselves), the speed (or x / y / z / path) graph with light on, movement onset/end and hit, and the video frame at light on / movement onset / movement end / trial end with the marker and the target switch. *Blocks & parts*: the groups of lights found (blocks = lights separated by the 30 s pauses) with the part each belongs to, taken in order from the plan; change a part or block number if needed (e.g. a repeated block) and Apply. Warnings if the count doesn't match the plan, a block doesn't have 30 lights, or the wrist/ankle was poorly visible.
   6. **Results** - the day's CSV + Excel and the all-participants Excel (below).
3. **Run all** (left, *Queue*) runs every day that still has steps to do, one after the other - e.g. add several participants and leave it overnight. A day that stops on a ✗ doesn't stop the others. *Stop* (or Esc) stops after the current part; finished steps are kept. From a terminal: `venv/bin/python pipeline.py` (all), `venv/bin/python pipeline.py S01` or `... S01 d2`.

Sessions made by the earlier version of the app (`results/<subject>/<session>/session.json`) show up as participants with their steps done.

**Trials.** Each light (in the 6-switch order, matched so a missed or extra light doesn't shift the rest) is a trial: start = light on; end = the button hit (the light goes off when the button is pressed) or, if not hit within the 1 s window, the light turning off by itself (**miss**). Trials are marked when the participant **started moving before the light** (the reach toward the target began 30-500 ms before the light: the speed rises continuously into the reach from before the light, toward the target) - these get a **negative reaction time** (the subject knew the next light of the sequence) - when the movement **didn't start from the home position** (*Start* column: *home*, *switch k* = it started at that switch, e.g. a shortcut from the previous one without going back home, or *elsewhere*; home = where the hand/foot rests when the lights come on in that block; not from home = more than halfway from home toward the target), and when the participant **first went to another switch and then corrected** (the first movement ended at another switch and the right one was hit afterwards; not a failure - marked so you can decide).

**Lights no camera sees** (e.g. an LED facing away): the next light always comes on ~1 s after a trial ends, so a gap with room for a trial is a hidden light - start = previous trial's end + 1 s, end = next light - 1 s (or the moment the hand/foot reaches the switch, at the end of a block); the switch = the one the hand/foot is on when it ends. These trials have *Light_source* = timing (video = the light was seen).

**Two cameras, one stops early** (or starts late): the session timeline covers both videos; where only one camera recorded, only that one is used. Each trial uses 3D when both cameras see the marker, otherwise the camera that sees it best (column *Data*).

**Measures** - the same method as the previous MATLAB system (`calculateMeasures.m`): the marker (wrist for the upper limb, ankle for the lower limb; 3D with two calibrated cameras, otherwise its position in the camera image) is resampled to 100 Hz, filtered (Butterworth 2nd order, 10 Hz, zero-phase) and its speed computed; movement onset = last sample before the first speed peak after the light with speed < 5% of the peak (reaction time = onset - light on); if the speed never drops below 5% after the light (the resting hand drifting slightly, or the end of the return home - the old code then gave RT ~ 0), the same 5% is measured from the resting level (lowest speed between the light and the peak); if the reach started before the light, the onset is that earlier moment and the reaction time is negative, movement end = first sample after the peak below 5% (movement time = end - onset); also AT, DT, AT/DT, number of speed peaks, straight-line deviation, end-point distance. One-frame MediaPipe glitches are removed first.

**Outputs** (in `results/<participant>/<day>/`, day = `d1` or `d2`):
- `<subject>_<session>_trials.csv` - every trial synced with the design: Part, Block, Sequence, Trial, Target, Limb, Side, Light_on_ms, Touch_ms (hand/foot reaches the switch in the video), Hit_ms, Light_off_ms, Trial_end_ms, Outcome (hit / miss / not detected), Anticipation (+ ms before the light), Not_from_home / Started_from / Start_distance_pct, Corrected (+ the switch it went to first), RT_s, MT_s.
- `<subject>_<session>_analysis.xlsx` - sheet *measures* (one row per trial: responsetime, movementtime, AT, DT, ..., touse) and the summary sheets of the old `makeCSVall.m`: mean / median / sd / iqr by part, by block and by sequence (familiarization and touse = 0 left out), plus *info*.
- `results/analysis_all_subjects.xlsx` - every analysed day together (made after each day's results, or with *Make the all-participants Excel*).

`results/` holds participant data and is not uploaded to git.

**How the switch lights are measured.** The pink LED light and its glow inside the box enlarged by 25% (the deep-red button, white table and skin are left out), minus its slowly changing background. The boxes follow camera movement automatically. A light partly hidden by the hand or foot still counts as on. A switch whose light stays on the whole video (e.g. IMG_0119, IMG_7554) has no light events.

Pressing the button turns the light off, so the touch is usually 0-3 frames (0-50 ms) before the light goes off. Times are video time in ms; at 60 fps they are accurate to one frame (~17 ms).

## Two GoPro cameras (recommended setup)
**Camera settings** - the same on both cameras: GoPro HERO13 Black, **4K, 60 fps, lens Linear, HyperSmooth (stabilization) OFF**, on fixed mounts (don't move them during the day - then one calibration serves every recording of that day).
- **Frame rate in Israel**: GoPro's "60 fps" is really 59.94 fps (Anti-Flicker 60 Hz). Mains power here is 50 Hz, so room lights flicker at 100 Hz; with **Anti-Flicker 50 Hz** the camera records **50 fps** (or 100 / 200), which avoids the flicker and gives a whole frame rate, so the timecode offset is exact. At 59.94 fps the timecode can be read as frames or as clock time (0.1 % apart, e.g. 30 ms for a 30 s offset).
- **Bit depth 8-bit** ("10-bit color" off): 10-bit HEVC decodes more slowly and the app uses 8 bits anyway. Codec HEVC (`GX` files) or H.264 (`GH` files) both work.
- The app shows each video's resolution, frame rate and codec, and a note when these settings are not as above.
- The timecode sync holds as long as the camera's clock runs; the clocks of two cameras drift apart slowly, so sync again at the start of each recording day and after a battery has been taken out.

**The board** - `venv/bin/python calibrate_cameras.py make-board` writes `outputs/charuco_board.pdf` (exact A4 scale) and `charuco_board.png`: a ChArUco board of 5 x 7 squares, 38 mm, `DICT_4X4_50` markers. Print the PDF **at 100% / Actual size** (not "fit to page"), check the 100 mm bar and one square with a ruler (38 mm; note the measured size), and glue it to something flat and stiff.

**Each recording day:**
1. **Sync the timecode**: in the GoPro Quik app, *Sync Timecode* (both cameras scan the QR code) - every recording day, and again after a battery was taken out. Then each camera can start / stop on its own - the timecode tells how the files line up.
2. **Record**: start both cameras; hold the board where both see it and slowly tilt / turn / move it around the subject's area for ~20-30 s (filling about 10-20 % of the picture); then run the whole experiment without stopping the cameras.
3. **App**: choose the 2 videos of that day - each camera's **first chapter** (`GX01xxxx.MP4`). GoPro splits long recordings into ~4 GB chapters (`GX010123.MP4`, `GX020123.MP4`, ...); the following chapters of the same clip are found automatically and used as one continuous video (the app warns if they don't follow each other). The rest - sync, calibration, switches, processing, analysis, results - runs by itself.

**Camera placement** (for good 3D and MediaPipe): both cameras see the whole body (head, torso, both arms, the legs for the lower limb), from in front of the subject (not from behind), 60-90° apart, about 1.5-2 m high and tilted down 20-35°, on rigid mounts; no window behind the subject.

The timecode is read with `ffprobe` (FFmpeg) if installed - on a Mac `brew install ffmpeg` (Homebrew: https://brew.sh); on Windows from https://ffmpeg.org/download.html, added to PATH - otherwise straight from the MP4's timecode track (no FFmpeg needed). Timecodes HH:MM:SS:FF and drop-frame HH:MM:SS;FF are converted with the file's real frame rate (e.g. 59.94).

**Tests**: `venv/bin/python -m unittest discover -s tests -v` (the automatic steps and the queue, timecodes incl. drop-frame and 59.94, the offset's sign, chapters, HERO13 files (HEVC 10-bit, 50 fps), pairing of board frames with an offset, the printed board).

Tip for the camera placement: MediaPipe works best when it sees the person roughly from the front or side, upright. A camera straight above (like IMG_012x) sees the arm well but guesses hips/legs under the table.

## Setup on macOS
The app uses MediaPipe's older `solutions` API, which only exists up to MediaPipe 0.10.21, and that version needs **Python 3.12 or older** (newer MediaPipe for Python 3.13+ no longer has it). To (re)create the environment in this folder:
```
python3 -m pip install --user uv
~/Library/Python/3.13/bin/uv python install 3.12
~/Library/Python/3.13/bin/uv venv --seed -p 3.12 venv
venv/bin/pip install -r requirements.txt
```
(Adjust `3.13` in the uv path to your `python3 --version`.) The first time you double-click `Lab App.command`, macOS may ask to allow Terminal to run it; if it says the file can't be opened, right-click it → Open.
