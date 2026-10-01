#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pyworld as pw
import soundfile as sf

PARAMETER_DEFAULTS = {
    "base_f0": 220.0,
    "pulse_lpf": 3500.0,
    "voiced_lpf_shift": 25.0,
    "breath_ratio": 0.02,
    "breath_hpf": 500.0,
    "breath_lpf": 8000.0,
    "source_gain": 0.0,
    "vtlen": 15.0,
    "f1_shift": 1.0,
    "f2_shift": 1.0,
    "f3_shift": 1.0,
    "f4_shift": 1.0,
    "q1": 5.0,
    "q2": 7.0,
    "q3": 20.0,
    "q4": 20.0,
    "qn": 10.0,
    "dip_freq": 7000.0,
    "dip_gain": -20.0,
    "dip_q": 8.0,
    "unvoiced_gain": 0.0,
    "unvoiced_lpf": 1.0,
    "unvoiced_hpf": 1.0,
    "burst_gain": 0.0,
    "burst_freq": 1.0,
}
ALL_PARAMETER_NAMES = tuple(PARAMETER_DEFAULTS)

VOICELESS_FRICATIVE_SYMBOLS = frozenset(["s", "sh", "h", "hy", "f"])
SPECIAL_PHONEMES = frozenset(["っ", "、", "。"])
SILENT_PARAMETER_NAMES = frozenset(["source_gain", "unvoiced_gain", "burst_gain"])


class KeyFrames:
    """キーフレーム"""

    def __init__(self, keyframes=None):
        # キーフレーム一覧
        self._keyframes = list(keyframes or [])

        # 補間用一時データ
        self._update_temporary = True
        self._x = np.zeros(0)
        self._y = np.zeros(0)

    @classmethod
    def constant(cls, value):
        return cls([[0.0, value]])

    @classmethod
    def from_line(cls, source, frame_period, initial_value):
        points = {0.0: float(initial_value)}

        # PureDataのvline~形式を読み込む
        for segment in source.split(","):
            fields = segment.split()
            if not 1 <= len(fields) <= 3:
                raise ValueError(f"不正なパラメータ列です: {segment.strip()}")

            # 各値を取得
            target = float(fields[0])
            duration = float(fields[1]) if len(fields) >= 2 else 0.0
            delay = float(fields[2]) if len(fields) >= 3 else 0.0

            # 開始時刻以降の折れ線を置換
            times = np.array(sorted(points))
            values = np.array([points[t] for t in times])
            value = float(np.interp(delay, times, values))
            points = {t: v for t, v in points.items() if t <= delay}
            if duration == 0.0:
                points[delay] = target
            else:
                points[delay] = value
                points[delay + duration] = target

        return cls([[time / frame_period, points[time]] for time in sorted(points)])

    def add(self, frame_index, value):
        # キーフレームを登録する
        self._keyframes.append((frame_index, value))

        # 補間用一時データを更新させる
        self._update_temporary = True

    def at(self, frame_index):
        # インデックス順に並べ替える
        if self._update_temporary:
            self._update_temporary = False
            x = np.array([i for i, _ in self._keyframes])
            y = np.array([v for _, v in self._keyframes])
            sorted_index = np.argsort(x)
            self._x = x[sorted_index]
            self._y = y[sorted_index]

        # 補間値を返す
        return np.interp(frame_index, self._x, self._y)

    def items(self):
        return list(self._keyframes)

    def map(self, func):
        return KeyFrames([[func(frame), value] for frame, value in self._keyframes])

    def trim_before(self, frame):
        return KeyFrames([[x, value] for x, value in self._keyframes if x < frame])


class ParameterFile:
    """話者・音素パラメータ"""

    def __init__(self, frame_period):
        self.frame_period = frame_period
        self.voice_definition = {"parameters": {}}
        self.phoneme_definitions = {}

    @classmethod
    def load(cls, path, frame_period):
        params = cls(frame_period=frame_period)

        # テキストファイルから話者定義・音素定義を読み込む
        with open(path, "r", encoding="utf-8") as f:
            current_section = None
            for linenum, line in enumerate(f, 1):
                # 行を取り出す
                line = line.strip().lstrip("\ufeff")
                if not line:
                    continue
                line = line.removesuffix(";").strip()

                # 行の先頭の単語を取得
                fields = line.split()
                name = fields[0]

                if name in ALL_PARAMETER_NAMES:
                    # パラメータ行
                    if current_section is None:
                        raise ValueError(f"{path}:{linenum}: 定義名より前にパラメータがあります")

                    # 記載されているパラメータの値を取得
                    values = line[len(name) :].strip()
                    if not values:
                        raise ValueError(f"{path}:{linenum}: {name}の値がありません")

                    # キーフレームを作成
                    initial_value = PARAMETER_DEFAULTS.get(name, 1.0)
                    keyframes = KeyFrames.from_line(
                        values, params.frame_period, initial_value
                    )
                    current_section["parameters"][name] = keyframes

                elif name == "voice_definition":
                    # 話者定義
                    current_section = params.voice_definition

                else:
                    # 音素定義
                    if len(fields) < 3:
                        raise ValueError(f"{path}:{linenum}: 音素定義にはVOTが必要です")
                    if name in params.phoneme_definitions:
                        raise ValueError(f"{path}:{linenum}: 音素 {name} は重複しています")
                    try:
                        vot = float(fields[-1])
                    except ValueError as e:
                        raise ValueError(f"{path}:{linenum}: VOTには数値を指定してください") from e
                    current_section = {
                        "symbols": fields[1:-1],
                        "vot": vot,
                        "parameters": {},
                    }
                    params.phoneme_definitions[name] = current_section

        return params


class Sequence:
    """音声合成用シーケンス"""

    def __init__(self, frame_period, fs, n_bins, f0_floor, frame_end, vot_frame):
        self.frame_begin = 0
        self.frame_end = frame_end
        self.frame_period = frame_period
        self.fs = fs
        self.n_bins = n_bins
        self.f0_floor = f0_floor
        self.vot_frame = vot_frame
        self.keyframes = {}

    def copy_with(self, frame_end=0, vot_frame=0.0):
        return Sequence(
            frame_period=self.frame_period,
            fs=self.fs,
            n_bins=self.n_bins,
            f0_floor=self.f0_floor,
            frame_end=frame_end,
            vot_frame=vot_frame,
        )

    def frame_values(self, frame_index):
        # キーフレームから1フレーム分の値を取り出す
        return {
            name: frames.at(frame_index)
            for name, frames in self.keyframes.items()
        }

    @classmethod
    def from_phoneme(cls, phoneme, params, fs, n_bins, f0_floor):
        # 音素定義から単一音素のシーケンスを作成する
        if phoneme not in params.phoneme_definitions:
            raise KeyError(f"音素定義がありません: {phoneme}")

        # 空のシーケンスを作成
        definition = params.phoneme_definitions[phoneme]
        seq = cls(
            frame_period=params.frame_period,
            fs=fs,
            n_bins=n_bins,
            f0_floor=f0_floor,
            frame_end=0.0,
            vot_frame=definition["vot"] / params.frame_period,
        )

        # シーケンスに各キーフレームを設定
        sources = params.voice_definition["parameters"] | definition["parameters"]
        for name in ALL_PARAMETER_NAMES:
            if name in sources:
                seq.keyframes[name] = sources[name]
            elif name in PARAMETER_DEFAULTS:
                seq.keyframes[name] = KeyFrames.constant(PARAMETER_DEFAULTS[name])
            else:
                raise ValueError(f"音素 {phoneme}: 定義に {name} が含まれていません")

        # 最終フレームを探索して設定
        last_frame = max(
            frame
            for name in ALL_PARAMETER_NAMES
            for frame, _ in seq.keyframes[name].items()
        )
        seq.frame_end = int(np.ceil(last_frame)) + 1

        return seq

    @classmethod
    def from_phoneme_list(
        cls,
        phoneme_list,
        params,
        fs=48000,
        n_bins=1025,
        f0_floor=20.0,
        vot_interval=200.0,
        comma_pause=200.0,
        period_pause=400.0,
    ):
        # 音素名リストをもとに音素を配置する
        pause_frames = {
            "、": comma_pause / params.frame_period,
            "。": period_pause / params.frame_period,
        }
        vot_interval_frames = vot_interval / params.frame_period
        current_frame = 0.0

        segments = []
        reference_seq = None
        silence_values = None
        sokuon_frame = None
        after_pause = False
        ends_with_phoneme = False

        # 音素セグメントを作成
        for index, phoneme in enumerate(phoneme_list):
            # 句読点の場合
            if phoneme in pause_frames:
                if sokuon_frame is not None:
                    raise ValueError("促音の直後に句読点は指定できません")

                # 句読点の配置を処理
                if reference_seq is None:
                    continue
                if not any(
                    p not in SPECIAL_PHONEMES
                    for p in phoneme_list[index + 1:]
                ):
                    continue
                pause_frame = pause_frames[phoneme]
                if pause_frame > 0:
                    silence = reference_seq.copy_with(
                        frame_end=max(int(np.ceil(pause_frame)), 1)
                    )
                    for name, value in silence_values.items():
                        silence.keyframes[name] = KeyFrames.constant(value)
                    segments.append((silence, current_frame))
                    current_frame += pause_frame
                after_pause = True
                ends_with_phoneme = False
                continue

            # 促音の場合
            if phoneme == "っ":
                if not ends_with_phoneme:
                    raise ValueError("促音の前に音素がありません")
                if (
                    index + 1 >= len(phoneme_list)
                    or phoneme_list[index + 1] in SPECIAL_PHONEMES
                ):
                    raise ValueError("促音の後に子音を含む音素がありません")

                # 促音の後続音素を確認
                next_phoneme = phoneme_list[index + 1]
                if next_phoneme not in params.phoneme_definitions:
                    raise KeyError(f"音素定義がありません: {next_phoneme}")
                next_symbols = params.phoneme_definitions[next_phoneme]["symbols"]
                if not next_symbols or next_symbols[0] in ("-", "N"):
                    raise ValueError("促音の後に子音を含む音素がありません")
                sokuon_frame = current_frame
                current_frame += vot_interval_frames
                ends_with_phoneme = False
                continue

            # 通常音素の場合
            seq = cls.from_phoneme(phoneme, params, fs, n_bins, f0_floor)

            # 基準シーケンスを設定
            if reference_seq is None:
                reference_seq = seq
                silence_values = {
                    name: (
                        0.0
                        if name in SILENT_PARAMETER_NAMES
                        else reference_seq.keyframes[name].items()[0][1]
                    )
                    for name in ALL_PARAMETER_NAMES
                }

            # 促音に続く音素を配置
            if sokuon_frame is not None:
                symbols = params.phoneme_definitions[phoneme]["symbols"]
                if symbols and symbols[0] in VOICELESS_FRICATIVE_SYMBOLS:
                    # 無声摩擦音の子音区間を伸長
                    if seq.vot_frame <= 0:
                        raise ValueError("VOT以前の区間がないため促音化できません")
                    if vot_interval_frames < seq.vot_frame:
                        raise ValueError("促音の長さが元の子音区間より短くなっています")
                    scale = vot_interval_frames / seq.vot_frame
                    shift = vot_interval_frames - seq.vot_frame
                    stretched = seq.copy_with(
                        frame_end=int(np.ceil(seq.frame_end + shift)),
                        vot_frame=vot_interval_frames,
                    )
                    for name in ALL_PARAMETER_NAMES:
                        stretched.keyframes[name] = seq.keyframes[name].map(
                            lambda frame: (
                                frame * scale
                                if frame <= seq.vot_frame
                                else frame + shift
                            )
                        )
                    seq = stretched
                    frame_offset = sokuon_frame
                else:
                    # 閉鎖区間の無音を追加
                    frame_offset = current_frame - seq.vot_frame
                    closure_frames = frame_offset - sokuon_frame
                    if closure_frames <= 0:
                        raise ValueError("VOT間隔が短すぎて促音を配置できません")
                    silence = reference_seq.copy_with(
                        frame_end=max(int(np.ceil(closure_frames)), 1)
                    )
                    for name, value in silence_values.items():
                        silence.keyframes[name] = KeyFrames.constant(value)
                    segments.append((silence, sokuon_frame))
                sokuon_frame = None
            elif after_pause:
                # 句読点直後の音素を配置
                frame_offset = current_frame
                current_frame += seq.vot_frame
                after_pause = False
            elif not segments:
                # 先頭音素を配置
                frame_offset = 0.0
                current_frame += seq.vot_frame
            else:
                # 通常音素をVOT基準で配置
                frame_offset = current_frame - seq.vot_frame

            # 通常音素を追加して次へ
            segments.append((seq, frame_offset))
            current_frame += vot_interval_frames
            ends_with_phoneme = True
        end_frame = current_frame

        if reference_seq is None:
            raise ValueError("発音する音素がありません")
        if sokuon_frame is not None:
            raise ValueError("促音の後に音素がありません")
        if any(
            next_offset <= offset
            for (_, offset), (_, next_offset) in zip(segments, segments[1:])
        ):
            raise ValueError("VOT間隔が短すぎて音素の開始順序を維持できません")

        # 音素セグメントを1本のシーケンスへ結合する
        seq = reference_seq._merge(segments, end_frame)

        # 音素末尾の音をフェードアウトさせる
        if ends_with_phoneme:
            tail_fade_frames = min(10.0, vot_interval) / params.frame_period
            fade_begin = end_frame - tail_fade_frames
            for name in SILENT_PARAMETER_NAMES:
                value = seq.keyframes[name].at(fade_begin)
                seq.keyframes[name] = seq.keyframes[name].trim_before(fade_begin)
                seq.keyframes[name].add(fade_begin, value)
                seq.keyframes[name].add(end_frame, 0.0)

        return seq

    def _merge(self, segments, end_frame):
        seq = self.copy_with(frame_end=int(np.ceil(end_frame)))

        # キーフレームを作成
        seq.keyframes = {name: KeyFrames() for name in ALL_PARAMETER_NAMES}
        for index, (segment, frame_offset) in enumerate(segments):
            # セグメントの有効範囲を決定
            if index < len(segments) - 1:
                next_frame_offset = segments[index + 1][1]
                local_frame_end = next_frame_offset - frame_offset
            else:
                next_frame_offset = None
                local_frame_end = end_frame - frame_offset

            # セグメントのキーフレームを転写
            for name in ALL_PARAMETER_NAMES:
                for frame, value in segment.keyframes[name].items():
                    if frame < local_frame_end:
                        seq.keyframes[name].add(frame + frame_offset, value)
                if next_frame_offset is not None:
                    value = segment.keyframes[name].at(local_frame_end)
                    seq.keyframes[name].add(
                        np.nextafter(next_frame_offset, frame_offset), value
                    )

        return seq


class FormantSynthesizer:
    """フォルマント合成器"""

    def __init__(self, sequence, master_gain=0.065, n_formants=9):
        if n_formants < 2:
            raise ValueError("n_formants must be at least 2")

        self.seq = sequence
        self.master_gain = master_gain
        self.n_formants = n_formants

    def create_freqs(self):
        # スペクトル処理用の周波数軸を作成
        n = self.seq.n_bins
        return np.arange(n) / (n - 1) * (self.seq.fs / 2)

    def _create_lpf_spectrum(self, freqs, f, q):
        s = 1j * freqs / f
        return 1 / (s * s + s / q + 1)

    def _create_bpf_spectrum(self, freqs, f, q):
        s = 1j * freqs / f
        return s / (s * s + s / q + 1)

    def _create_hpf_spectrum(self, freqs, f, q):
        s = 1j * freqs / f
        return s * s / (s * s + s / q + 1)

    def _create_peaking_spectrum(self, freqs, f, gain, q):
        s = 1j * freqs / f
        ss = s * s
        a = 10 ** (gain / 40)
        return (ss + s * a / q + 1) / (ss + s / (a * q) + 1)

    def synthesize_frame(self, frame_index, freqs):
        values = self.seq.frame_values(frame_index)

        # 基本周波数
        f0 = values["base_f0"]
        f0_clip = 0.0 if f0 < self.seq.f0_floor else f0
        f0_safe = max(f0_clip, 1.0)

        # 音源: 声帯波
        pulse_gain = values["source_gain"] * (1 - values["breath_ratio"])
        sp_pulse = self._create_lpf_spectrum(freqs, values["pulse_lpf"], 1) ** 2
        harmonic_index = np.maximum(freqs / f0_safe, 1.0)
        sp_saw = 2 / (np.pi * harmonic_index)
        sp_saw[0] = 0.0
        if f0_clip != 0:
            sp_pulse *= sp_saw
        sp_pulse *= max(pulse_gain, 0)

        # 音源: 吐息ノイズ
        breath_gain = values["source_gain"] * values["breath_ratio"] * 0.5
        sp_breath = self._create_hpf_spectrum(freqs, values["breath_hpf"], 1)
        sp_breath *= self._create_lpf_spectrum(freqs, values["breath_lpf"], 1)
        sp_breath *= max(breath_gain, 0)

        # フォルマント
        formant_base = 34000 / (4 * values["vtlen"])
        formant_shift = [
            values["f1_shift"],
            values["f2_shift"],
            values["f3_shift"],
            values["f4_shift"],
        ]
        formant_q = [values["q1"], values["q2"], values["q3"], values["q4"]]
        sp_formant = None
        for i in range(self.n_formants):
            formant = i + 1
            shift = formant_shift[i] if i < len(formant_shift) else 1
            f = formant_base * (2 * formant - 1) * shift
            q = formant_q[i] if i < len(formant_q) else values["qn"]

            if i == 0:
                sp_formant = self._create_lpf_spectrum(freqs, f, q)
            else:
                sign = -1 if i % 2 == 1 else 1
                sp_formant += sign * self._create_bpf_spectrum(freqs, f, q)

        # スペクトルディップ
        sp_dip = self._create_peaking_spectrum(
            freqs,
            values["dip_freq"],
            values["dip_gain"],
            values["dip_q"],
        )

        # 有声音
        voiced_lpf = formant_base * values["voiced_lpf_shift"]
        sp_voiced = sp_formant * self._create_lpf_spectrum(freqs, voiced_lpf, 1)
        sp_voiced *= sp_dip

        # 無声子音
        unvoiced_lpf = formant_base * values["unvoiced_lpf"]
        unvoiced_hpf = formant_base * values["unvoiced_hpf"]
        sp_unvoiced = self._create_hpf_spectrum(freqs, unvoiced_hpf, 1)
        sp_unvoiced *= self._create_lpf_spectrum(freqs, unvoiced_lpf, 1)
        sp_unvoiced *= max(values["unvoiced_gain"], 0)

        # 破裂音
        burst_freq = formant_base * values["burst_freq"]
        sp_burst = self._create_bpf_spectrum(freqs, burst_freq, 5)
        sp_burst *= max(values["burst_gain"], 0)

        # 周期成分と非周期成分を混合
        sp_v = np.abs(sp_voiced * sp_pulse)
        sp_u_power = (
            np.abs(sp_voiced * sp_breath) ** 2
            + np.abs(sp_unvoiced) ** 2
            + np.abs(sp_burst) ** 2
        )
        sp_u = np.sqrt(sp_u_power + 1e-12) - 1e-6

        # WORLD向けのゲイン補正
        gain_v = np.sqrt(self.seq.fs / f0_safe) / 2
        gain_u = 1 / np.sqrt(3)
        if f0_clip != 0:
            sp_v *= self.master_gain * gain_v
        sp_u *= self.master_gain * gain_u

        # WORLD特徴量を作成
        sp = sp_v**2 + sp_u**2
        ap = sp_u / np.sqrt(np.maximum(sp, 1e-24))
        ap = np.clip(ap, 1e-6, 1.0)

        return f0_clip, sp, ap

    def synthesize(self, frame_begin=None, frame_end=None):
        # 連番フレームで合成
        if frame_begin is None:
            frame_begin = self.seq.frame_begin
        if frame_end is None:
            frame_end = self.seq.frame_end

        n_frames = frame_end - frame_begin
        freqs = self.create_freqs()
        f0 = np.zeros(n_frames)
        sp = np.zeros((n_frames, self.seq.n_bins))
        ap = np.zeros((n_frames, self.seq.n_bins))

        # 各フレームごとに処理
        for i in range(n_frames):
            f0[i], sp[i, :], ap[i, :] = self.synthesize_frame(frame_begin + i, freqs)

        # ゼロを入れるとWORLDがnanを生成する場合があるので調整
        sp = np.maximum(sp, 1e-16)
        ap = np.maximum(ap, 1e-6)

        return f0, sp, ap


def main():
    # 話者定義・音素定義を読み込む
    infile = Path(__file__).with_name("fs4.txt")
    params = ParameterFile.load(infile, frame_period=5.0)

    # シーケンスを作成
    #p = "あ ら ゆ る げ ん じ つ お 、 す べ て じ ぶ ん の ほ お え ね じ ま げ た の だ".split()
    #p = "あ い う え お 。 か き  く け こ 。 さ し す せ そ 。 た ち つ て と 。 な に ぬ ね の 。 は ひ ふ へ ほ 。 ま み む め も 。 や ゆ よ 。 ら り る れ ろ 。 わ を ん 。 が ぎ ぐ げ ご 。 ざ じ ず ぜ ぞ 。 だ ぢ づ で ど 。 ば び ぶ べ ぼ 。 ぱ ぴ ぷ ぺ ぽ 。 きゃ きゅ きょ 。 しゃ しゅ しょ 。 ちゃ ちゅ ちょ。にゃ にゅ にょ。ひゃ ひゅ ひょ 。 みゃ みゅ みょ 。 りゃ りゅ りょ 。 ぎゃ ぎゅ ぎょ 。 じゃ じゅ じょ 。 びゃ びゅ びょ 。 ぴゃ ぴゅ ぴょ 。 いぇ 。 うぃ うぇ うぉ 。 くぁ くぃ くぇ くぉ 。 ぐぁ。 つぁ つぃ つぇ つぉ 。 てぃ でぃ てゅ でゅ 。 ふぁ ふぃ ふぇ ふぉ 。 ヴぁ ヴぃ ヴ ヴぇ ヴぉ 。 しぇ じぇ ちぇ 。 あっ 、 あー 。".split()
# KeyError: '音素定義がありません: を'
#    p = "あ い う え お 。 か き  く け こ 。 さ し す せ そ 。 た ち つ て と 。 な に ぬ ね の 。 は ひ ふ へ ほ 。 ま み む め も 。 や ゆ よ 。 ら り る れ ろ 。 わ お ん 。".split()
# KeyError: '音素定義がありません: ぢ'
#    p = "が ぎ ぐ げ ご 。 ざ じ ず ぜ ぞ 。 だ ぢ づ で ど 。 ば び ぶ べ ぼ 。 ぱ ぴ ぷ ぺ ぽ 。 きゃ きゅ きょ 。 しゃ しゅ しょ 。 ちゃ ちゅ ちょ。にゃ にゅ にょ。ひゃ ひゅ ひょ 。 みゃ みゅ みょ 。 りゃ りゅ りょ 。 ぎゃ ぎゅ ぎょ 。 じゃ じゅ じょ 。 びゃ びゅ びょ 。 ぴゃ ぴゅ ぴょ 。 いぇ 。 うぃ うぇ うぉ 。 くぁ くぃ くぇ くぉ 。 ぐぁ。 つぁ つぃ つぇ つぉ 。 てぃ でぃ てゅ でゅ 。 ふぁ ふぃ ふぇ ふぉ 。 ヴぁ ヴぃ ヴ ヴぇ ヴぉ 。 しぇ じぇ ちぇ 。 あっ 、 あー 。".split()
#KeyError: '音素定義がありません: ぢ'
#KeyError: '音素定義がありません: づ'
#KeyError: '音素定義がありません: くぁ'

#    p = "が ぎ ぐ げ ご 。 ざ じ ず ぜ ぞ 。 だ じ ず で ど 。 ば び ぶ べ ぼ 。 ぱ ぴ ぷ ぺ ぽ 。 きゃ きゅ きょ 。 しゃ しゅ しょ 。 ちゃ ちゅ ちょ 。 にゃ にゅ にょ 。 ひゃ ひゅ ひょ 。 みゃ みゅ みょ 。 りゃ りゅ りょ 。 ぎゃ ぎゅ ぎょ 。 じゃ じゅ じょ 。 びゃ びゅ びょ 。 ぴゃ ぴゅ ぴょ 。 いぇ 。 うぃ うぇ うぉ 。 つぁ つぃ つぇ つぉ 。 てぃ でぃ てゅ でゅ 。 ふぁ ふぃ ふぇ ふぉ 。 ちぇ 。 、 。".split()

# 不足している音素============================================================================
# を ぢ づ くぁ くぃ くぇ くぉ  ぐぁ  ヴぁ ヴぃ ヴ ヴぇ ヴぉ  しぇ じぇ っ  あ ー 
# ============================================================================================

    #p = "を ぢ づ くぁ くぃ くぇ くぉ  ぐぁ  ヴぁ ヴぃ ヴ ヴぇ ヴぉ  しぇ じぇ っ た ".split()
    p = "を 。 ぢ 。 づ 。 くぁ 。 くぃ 。 くぇ 。 くぉ 。  ぐぁ 。  ゔぁ 。 ゔぃ 。 ゔ 。 ゔぇ 。 ゔぉ 。 ゔぉ っ く す あ っ た ".split()

#    p = "が ぎ ぐ げ ご 。 ざ じ ず ぜ ぞ 。 だ じ ず で ど 。 ば び ぶ べ ぼ 。 ぱ ぴ ぷ ぺ ぽ 。 きゃ きゅ きょ 。 しゃ しゅ しょ 。 ちゃ ちゅ ちょ 。 にゃ にゅ にょ 。 ひゃ ひゅ ひょ 。 みゃ みゅ みょ 。 りゃ りゅ りょ 。 ぎゃ ぎゅ ぎょ 。 じゃ じゅ じょ 。 びゃ びゅ びょ 。 ぴゃ ぴゅ ぴょ 。 いぇ 。 うぃ うぇ うぉ 。 くぁ くぃ くぇ くぉ 。 ぐぁ。 つぁ つぃ つぇ つぉ 。 てぃ でぃ てゅ でゅ 。 ふぁ ふぃ ふぇ ふぉ 。 ヴぁ ヴぃ ヴ ヴぇ ヴぉ 。 しぇ じぇ ちぇ 。 あっ 、 あー 。".split()
# が ぎ ぐ げ ご 。 ざ じ ず ぜ ぞ 。 だ ぢ づ で ど 。 ば び ぶ べ ぼ 。 ぱ ぴ ぷ ぺ ぽ 。 きゃ きゅ きょ 。 しゃ しゅ しょ 。 ちゃ ちゅ ちょ。にゃ にゅ にょ。ひゃ ひゅ ひょ 。 みゃ みゅ みょ 。 りゃ りゅ りょ 。 ぎゃ ぎゅ ぎょ 。 じゃ じゅ じょ 。 びゃ びゅ びょ 。 ぴゃ ぴゅ ぴょ 。 いぇ 。 うぃ うぇ うぉ 。 くぁ くぃ くぇ くぉ 。 ぐぁ。 つぁ つぃ つぇ つぉ 。 てぃ でぃ てゅ でゅ 。 ふぁ ふぃ ふぇ ふぉ 。 ヴぁ ヴぃ ヴ ヴぇ ヴぉ 。 しぇ じぇ ちぇ 。 あっ 、 あー 。".split()
    seq = Sequence.from_phoneme_list(p, params)

    # 合成
    synth = FormantSynthesizer(seq)
    f0, sp, ap = synth.synthesize()
    y = pw.synthesize(f0, sp, ap, seq.fs, seq.frame_period)  # WORLDで合成

    # 保存
    sf.write("out.wav", y, seq.fs, "FLOAT")


if __name__ == "__main__":
    main()

