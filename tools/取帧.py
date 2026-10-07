#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""取帧 —— 定底稿前，先把「整个镜头」扫一遍，找出元素最全的那一帧。

【为什么要有这个工具】
  2026-10-05 案例 镜头02 翻车：一个 11 秒（6.0~17.0s）的镜头，我截了 t=7.5 当底稿，
  那一格棋子还全在左上暂存区、棋盘上光秃秃 —— 而「道路和村庄摆在哪」正是这一帧
  唯一该钉住的东西，一个都没钉住。元素最全的是 t≈16.0。
  两个错叠在一起：
    ① 取帧范围只铺到 t=11.5，而落子发生在第 12 秒之后 —— 正确的帧根本不在我扫的区间里；
    ② 挑选标准挑反了：我按「暂存区干净、没有红叉」挑，恰好挑走了信息最少的那一格。

【规则（用户 2026-10-05 定）】
  **默认取「这个镜头里元素最全的那一帧」当底稿。**
  镜头是动的 —— 可能是空背景→出地图→出人→出棋子道路→出字，也可能反过来。
  底稿是「改到最完善」的起点，起点不全，改出来的也不全。
  除非你**明确知道某一帧有病**，那才另外截那一帧（并写进清单的「已知问题帧」）。

【本工具的主产出 = 接片图，不是自动打分】
  实测：自动的「元素最全」打分在本项目**不可靠**，两条路都失败过 ——
    · 跟时间中位数求差：镜头02 把切点帧 t=6.0 排第一（96.02 分，第二名才 5.17），
      因为开场帧离中位数最远；正确答案 t=16.0 排第 12。**镜头内有相机推镜，也会污染这个量。**
    · 数「小连通域面积」：被相机远近主导，同样挑不出 t=16.0。
  根因：本镜头里"多出来的元素"（一段路 + 一座村庄）只占极小的像素面积，
  任何全图性的指标都会被**推镜的缩放**盖过去。这不是调参能救的。
  所以：**接片图（全镜头一屏看完）才是主产出，人眼定帧。**
  自动打分只作「提示」，附在图上，明确标了不可靠；它算的是"离基线多远"，**不等于元素多少**。

【用法】
  python3 取帧.py --视频 成片/块1_xxx.mp4 --起 6 --止 17 --步 0.5 --出 1005/审 --镜头 02
  --起/--止 必须铺满**整个镜头**（不许只截手顺的那一段）。

【产出】
  <出>/取帧_候选/<镜头>_t<秒>.jpg   —— 候选帧原图（底稿就从这里挑，路径直接粘进清单）
  <出>/取帧_接片_<镜头>.jpg        —— ★主产出：全镜头按时间顺序一屏看完
  <出>/取帧_提示_<镜头>.jpg        —— 副产出：自动打分曲线 + Top5（**仅供参考，不可靠**）
  终端：帧清单、首尾对照、自动提示 Top3（带免责声明）、可直接粘进清单的「取帧」模板

【不做什么】不删帧、不改原视频；只往 <出> 写文件。
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np


def 取字体(大小):
    from PIL import ImageFont
    for p in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc",
              "/System/Library/Fonts/Supplemental/Songti.ttc"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, 大小)
            except Exception:
                pass
    return ImageFont.load_default()


def 抽帧(视频, 起, 止, 步, 出):
    """按步长抽帧，返回 [(秒, 路径)]。"""
    os.makedirs(出, exist_ok=True)
    out = []
    # 用整数计数避免浮点累加漂移
    n = int(round((止 - 起) / 步)) + 1
    for i in range(n):
        t = round(起 + i * 步, 3)
        p = os.path.join(出, "t%.1f.jpg" % t)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", "%.3f" % t, "-i", 视频,
             "-frames:v", "1", "-q:v", "2", "-y", p],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode != 0 or not os.path.exists(p):
            print("！抽帧失败 t=%.1f：%s" % (t, r.stderr.decode()[:200]))
            continue
        out.append((t, p))
    return out


def 缩略(帧表, 工作宽=360):
    """统一缩到工作宽，返回 [(秒, 原路径, BGR 小图)]。"""
    小图 = []
    for t, p in 帧表:
        im = cv2.imread(p)
        if im is None:
            print("！读不到 %s" % p)
            continue
        h = max(1, int(round(im.shape[0] * 工作宽 / im.shape[1])))
        小图.append((t, p, cv2.resize(im, (工作宽, h), interpolation=cv2.INTER_AREA)))
    return 小图


def 打分(小图):
    """自动提示分：每帧跟「时间中位数帧」的平均逐像素差。

    **这不是「元素多少」**，是「离这条时间线的基线背景多远」。镜头一动就失真，
    切点帧会冲到第一。只当提示用。返回 [(秒, 路径, 分, 占比, 小图)]，分降序。
    """
    if not 小图:
        return []
    堆 = np.stack([s for _, _, s in 小图]).astype(np.float32)
    中位 = np.median(堆, axis=0)
    项 = []
    for t, p, s in 小图:
        差 = np.abs(s.astype(np.float32) - 中位).max(axis=2)
        项.append((t, p, float(差.mean()), float((差 > 25).mean() * 100.0), s))
    项.sort(key=lambda e: -e[2])
    return 项


def 接片(小图, 出, 镜头, 列=6, 格宽=300):
    """★主产出：全镜头按时间顺序排成一张大图，一屏看完，人眼挑最全的那一格。"""
    from PIL import Image, ImageDraw
    序 = sorted(小图, key=lambda e: e[0])
    if not 序:
        return None
    行 = (len(序) + 列 - 1) // 列
    比 = 序[0][2].shape[0] / float(序[0][2].shape[1])
    格高 = int(round(格宽 * 比))
    边, 标注 = 8, 24
    头高 = 44
    宽 = 边 + 列 * (格宽 + 边)
    高 = 头高 + 行 * (格高 + 标注 + 边)
    画 = Image.new("RGB", (宽, 高), (26, 26, 26))
    d = ImageDraw.Draw(画)
    d.text((10, 6), "全镜头接片 镜头%s   %.1f~%.1fs 每 %.1fs 一帧   "
                    "左→右、上→下＝时间顺序" % (镜头, 序[0][0], 序[-1][0],
                    round(序[1][0] - 序[0][0], 3) if len(序) > 1 else 0),
           font=取字体(24), fill=(120, 220, 255))
    d.text((10, 30), "底稿从这里挑：哪一格把本镜该有的元素摆得最全，就挑哪一格"
                     "（不一定是最靠后的）", font=取字体(18), fill=(170, 170, 170))
    字 = 取字体(20)
    for i, (t, _, s) in enumerate(序):
        r, c = divmod(i, 列)
        x = 边 + c * (格宽 + 边)
        y = 头高 + r * (格高 + 标注 + 边)
        图 = Image.fromarray(cv2.cvtColor(
            cv2.resize(s, (格宽, 格高), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        画.paste(图, (x, y))
        d.rectangle([x - 1, y - 1, x + 格宽, y + 格高], outline=(70, 70, 70))
        d.text((x + 2, y + 格高 + 3), "t=%.1fs" % t, font=字, fill=(235, 235, 235))
    p = os.path.join(出, "取帧_接片_%s.jpg" % 镜头)
    画.save(p, quality=92)
    return p


def 提示图(项, 出, 镜头, 全宽=1180):
    """副产出：自动打分曲线（按时间）+ Top5 缩略图。**明确标注不可靠。**"""
    from PIL import Image
    曲线高 = 300
    左宽 = 560
    画 = np.full((曲线高 + 44, 全宽, 3), 24, np.uint8)
    序 = sorted(项, key=lambda e: e[0])
    ts = [e[0] for e in 序]
    分 = [e[2] for e in 序]
    lo, hi = min(ts), max(ts)
    mx = max(分) * 1.05 if 分 and max(分) > 0 else 1.0
    if hi == lo:
        hi = lo + 1
    cv2.putText(画, "auto hint: dist-from-median  vs  time (s)  -- UNRELIABLE",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (120, 220, 255), 2)
    cv2.putText(画, "camera push-in / cut frames dominate this; do NOT trust the peak",
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 220), 1)
    cv2.rectangle(画, (50, 58), (左宽, 58 + 曲线高 - 30), (60, 60, 60), 1)

    def 点(t, v):
        x = int(50 + (t - lo) / (hi - lo) * (左宽 - 60))
        y = int(58 + (曲线高 - 30) * (1 - v / mx))
        return x, y

    for a, b in zip(序[:-1], 序[1:]):
        cv2.line(画, 点(a[0], a[2]), 点(b[0], b[2]), (80, 200, 255), 2)
    for e in 序:
        cv2.circle(画, 点(e[0], e[2]), 3, (80, 200, 255), -1)
    峰 = 项[0][0]
    x, y = 点(峰, 项[0][2])
    cv2.circle(画, (x, y), 7, (60, 60, 235), -1)
    cv2.putText(画, "peak t=%.1f (hint only)" % 峰,
                (min(x + 10, 左宽 - 170), max(y - 12, 74)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 235), 2)
    for 秒 in 序[:: max(1, len(序) // 8)]:
        x, _ = 点(秒[0], 0)
        cv2.putText(画, "%.0f" % 秒[0], (x - 10, 58 + 曲线高 - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    x0 = 左宽 + 20
    图宽 = (全宽 - x0 - 10 * 5) // 5
    for i, e in enumerate(项[:5]):
        s = e[4]
        k = 图宽 / s.shape[1]
        s = cv2.resize(s, (图宽, int(round(s.shape[0] * k))), interpolation=cv2.INTER_AREA)
        x = x0 + i * (图宽 + 10)
        y = 58
        画[y:y + s.shape[0], x:x + s.shape[1]] = s
        cv2.putText(画, "#%d t=%.1f" % (i + 1, e[0]), (x + 2, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 120) if i == 0 else (200, 200, 200), 2)
        cv2.putText(画, "%.1f" % e[2], (x + 2, y + s.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    p = os.path.join(出, "取帧_提示_%s.jpg" % 镜头)
    Image.fromarray(cv2.cvtColor(画, cv2.COLOR_BGR2RGB)).save(p, quality=93)
    return p


def 主():
    ap = argparse.ArgumentParser()
    ap.add_argument("--视频", required=True)
    ap.add_argument("--起", type=float, required=True)
    ap.add_argument("--止", type=float, required=True)
    ap.add_argument("--步", type=float, default=0.5)
    ap.add_argument("--出", required=True)
    ap.add_argument("--镜头", default="NN")
    ap.add_argument("--列", type=int, default=6, help="接片图列数")
    a = ap.parse_args()

    if not os.path.exists(a.视频):
        print("！找不到视频 %s" % a.视频)
        return 1
    if a.止 <= a.起:
        print("！--止 必须大于 --起")
        return 1

    候选目录 = os.path.join(a.出, "取帧_候选")
    帧表 = 抽帧(a.视频, a.起, a.止, a.步, 候选目录)
    print("抽到 %d 帧：%.1f~%.1fs 每 %.1fs 一帧" % (len(帧表), a.起, a.止, a.步))
    if not 帧表:
        print("！一帧都没抽到")
        return 1
    if abs(帧表[0][0] - a.起) > 1e-6 or abs(帧表[-1][0] - a.止) > 1e-6:
        print("！抽帧范围没铺满 %.1f~%.1fs，实际抽到 %.1f~%.1fs"
              % (a.起, a.止, 帧表[0][0], 帧表[-1][0]))

    小图 = 缩略(帧表)
    项 = 打分(小图)

    # 把候选帧改成带镜头号的名字，方便直接把路径粘进清单
    for t, p, _ in 小图:
        新 = os.path.join(候选目录, "%s_t%.1f.jpg" % (a.镜头, t))
        if os.path.abspath(p) != os.path.abspath(新):
            os.replace(p, 新)
    小图 = [(t, os.path.join(候选目录, "%s_t%.1f.jpg" % (a.镜头, t)), s) for t, _, s in 小图]
    项 = [(t, os.path.join(候选目录, "%s_t%.1f.jpg" % (a.镜头, t)), d, r, s)
          for t, _, d, r, s in 项]
    项.sort(key=lambda e: -e[2])

    # ---- 首尾对照：空↔满型镜头，最全帧大概率在其中一端 ----
    首, 尾 = 小图[0], 小图[-1]
    首尾差 = float(np.abs(首[2].astype(np.float32) - 尾[2].astype(np.float32)).mean())

    print("\n【自动提示】按「离基线远近」排 —— 不可靠，镜头一动就失真，只当提示：")
    for i, (t, _, d, r, _) in enumerate(项[:5]):
        print("   #%d  t=%5.1fs   分 %6.2f   占比 %5.2f%%" % (i + 1, t, d, r))
    print("  （镜头02 实测：这个打分把切点帧 t=6.0 排第一，正确答案 t=16.0 排第 12。别信它。）")
    print("\n【首尾对照】首帧 t=%.1f 与末帧 t=%.1f 的平均差 = %.2f" % (首[0], 尾[0], 首尾差))
    if 首尾差 >= 25.0:
        print("  ⚠ 首尾差极大：**首帧很可能落在转场/切点上**（镜头02 的 t=6.0 就是，"
              "它离基线 96 分、第二名才 5.17）。转场帧糊、带前一镜的尾巴，**不能当底稿**。"
              "先把转场帧排除掉，再在剩下的帧里挑最全的。")
    elif 首尾差 >= 3.0:
        print("  → 差距明显：本镜是「空↔满」型，最全帧**大概率在其中一端**。"
              "打开接片图，先看头两格和尾两格。")
    else:
        print("  → 差距不大：本镜可能全程元素差不多，或变化发生在中段。**接片图逐格看。**")

    p1 = 接片(小图, a.出, a.镜头, 列=a.列)
    p2 = 提示图(项, a.出, a.镜头)
    print("\n★ 主产出（**打开它，人眼定帧**）：%s" % p1)
    print("  副产出（自动提示，别信峰值）：  %s" % p2)

    候选 = os.path.join(候选目录, "%s_t<秒>.jpg" % a.镜头)
    print("""
──────── 挑完帧，把这段填进 镜头清单.json 的 镜头["%s"] 里（改图.py 关卡 H 会查）────────
  "取帧": {
    "全镜头区间": [%.1f, %.1f],
    "取帧命令": "python3 ~/.claude/skills/MG动画/tools/取帧.py --视频 %s --起 %.1f --止 %.1f --步 %.1f --出 %s --镜头 %s",
    "元素最全帧": "%s",
    "底稿": "%s",
    "取帧依据": "（一句话：这一格比别的格多了什么，所以它最全）",
    "已知问题帧": [],
    "例外理由": ""
  }
  规则：底稿默认 = 元素最全帧；两者不一样就必须在「例外理由」里说清楚，并把有病的帧写进「已知问题帧」。
────────────────────────────────────────────────────────────""" % (
        a.镜头, a.起, a.止, a.视频, a.起, a.止, a.步, a.出, a.镜头, 候选, 候选))
    return 0


if __name__ == "__main__":
    sys.exit(主())
