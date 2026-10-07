# -*- coding: utf-8 -*-
"""比对.py —— 改图后 / 生视频前的核验关卡。

它回答两个问题（都是 2026-09-30 用户定规的）：
  **A. 改完图，是不是"只有该改的地方变了"？**         → `--底稿`
  **B. 手上这几帧，是不是同一个版本 / 同一个风格？**   → `--两两`

为什么要能量化：肉眼看"差不多"，实测能差 5~20 个色阶。
判据来自"底稿自己相邻帧的抖动"——那就是"没被改"的噪声水平，
比它大一个量级，就是被重画了。**这个基线每个项目要自己测一遍，不许照抄。**

🔴 判据常量有三个，**全部逐项目设定，不许照抄**（2026-10-07 修正）：
   ① 抖动基线  ② 画幅/尺寸  ③ 背景区采样框
   2026-10-07 之前 ②③ 是写死的竖版 720×1280 + 案例版图的上下两条，
   结果换到 16:9 的项目时，图会被**拉伸变形**、采样框落在无意义的位置，
   §8.0 的「三查」会给出一个**看起来正常、其实毫无意义**的数字 —— 不报错，只给假结果。
   现在：尺寸默认**取输入图的原始尺寸**（不做任何缩放），
        背景区默认按 `--画幅` 选预设，两者都可用参数显式覆盖。

用法：
    # A. 改图后核验：出图和底稿比（尺寸默认 = 底稿原尺寸）
    python3 比对.py --底稿 体检/块1v3/镜头02.jpg --出图 改图出图/xx/镜头02_第1张.png ...

    # B. 生视频前核验：手上要送进去的几帧两两互比（判断是否同版本/同风格）
    python3 比对.py --两两 帧A.png 帧B.png 帧C.png ...

    # 换项目要显式声明画幅（16:9 的片子务必写，否则采样框是竖版那套）
    python3 比对.py --画幅 16:9 --两两 F0.png F30.png F60.png

    # 采样框不对（比如画面上下不是纯背景）就手动给，分号隔开多块：
    python3 比对.py --背景区 "0.03,0.04,0.97,0.13;0.03,0.87,0.97,0.96" --底稿 A --出图 B

输出：终端数值表 + <输出目录>/_比对_总览.jpg（原图 / 出图 / 差异热力图，并排）
"""
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

# —— 判据常量（默认值，可被命令行覆盖）——
尺寸 = None        # None = 用第一张图的**原始尺寸**，绝不缩放变形
背景区 = None      # None = 按 画幅 取下面的预设
抖动基线 = 3.0
画幅 = "9:16"

# 背景区预设：**画面里不该被改动的纯背景区**当探针。
# 只是通用默认（上下两条窄边）——**每个项目必须打开图确认它真的是纯背景**，
# 不是的话用 `--背景区` 手动指定。错采样框 = 假结论。
画幅预设 = {
    "16:9": [(0.03, 0.04, 0.97, 0.13),      # 上边一条
             (0.03, 0.87, 0.97, 0.96)],     # 下边一条
    "9:16": [(0.0833, 0.0469, 0.9167, 0.1484),
             (0.0833, 0.9219, 0.9167, 0.9844)],
}


def 解析尺寸(文本):
    """'1280x720' / '1280×720' / '1280,720' → (1280, 720)"""
    数 = [int(v) for v in re.split(r"[xX×,]\s*", 文本.strip()) if v.strip()]
    if len(数) != 2:
        raise ValueError("--尺寸 要写成 宽x高，例如 1280x720（给的是 %r）" % 文本)
    return (数[0], 数[1])


def 解析背景区(文本):
    """'x0,y0,x1,y1;x0,y0,x1,y1' → [(...), (...)]（比例值，0~1）"""
    块 = []
    for 段 in 文本.split(";"):
        段 = 段.strip()
        if not 段:
            continue
        数 = [float(v) for v in 段.split(",")]
        if len(数) != 4:
            raise ValueError("每个采样块要写四个比例值 x0,y0,x1,y1（给的是 %r）" % 段)
        块.append(tuple(数))
    if not 块:
        raise ValueError("--背景区 是空的")
    return 块


def 定配置(args, 首图):
    """把命令行 / 输入图定出来的三个常量写进全局。返回要打印的告知行。"""
    global 尺寸, 背景区, 抖动基线, 画幅
    告知 = []

    if args.尺寸:
        尺寸 = 解析尺寸(args.尺寸)
        告知.append("尺寸：%dx%d（命令行指定）" % 尺寸)
    else:
        尺寸 = (首图.width, 首图.height)
        告知.append("尺寸：%dx%d（取输入图原始尺寸，不缩放）" % 尺寸)

    if args.背景区:
        背景区 = 解析背景区(args.背景区)
        告知.append("背景区：命令行指定 %d 块" % len(背景区))
    else:
        画幅 = args.画幅
        背景区 = list(画幅预设.get(画幅, 画幅预设["9:16"]))
        告知.append("背景区：按 --画幅 %s 取预设 %d 块" % (画幅, len(背景区)))

    抖动基线 = float(args.抖动基线) if args.抖动基线 is not None else 3.0
    告知.append("抖动基线：%.1f%s"
              % (抖动基线,
                 "（命令行指定）" if args.抖动基线 is not None
                 else "（默认 3.0 —— 逐项目应实测后传 --抖动基线）"))
    return 告知


def 载入(p, 尺寸):
    img = Image.open(p).convert("RGB")
    if (img.width, img.height) != 尺寸:
        img = img.resize(尺寸)
    return np.asarray(img, dtype=np.float32)


def 背景均值(a):
    H, W = a.shape[:2]
    块 = [a[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)].reshape(-1, 3)
          for x0, y0, x1, y1 in 背景区]
    return np.concatenate(块).mean(axis=0)


def 差异图(a, b):
    if a.shape != b.shape:
        # 尺寸不一时按小的一方裁 —— 早该在 定配置 里统一，这里只是兜底不崩
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    return np.abs(a - b).mean(axis=2)


def 判定(色差):
    if 色差 <= 抖动基线:
        return "锁住"
    if 色差 < 10:
        return "轻微"
    return "★被重画"


def 模式_对比(底稿路径, 出图们, 输出目录):
    首 = Image.open(底稿路径).convert("RGB")
    告知 = 定配置(main.args, 首)
    for 行 in 告知:
        print(行)
    print()

    底 = np.asarray(首 if (首.width, 首.height) == 尺寸 else 首.resize(尺寸), dtype=np.float32)
    w0 = 背景均值(底)
    print("底稿（%s）背景区平均色  RGB(%5.1f,%5.1f,%5.1f)" % (os.path.basename(底稿路径), *w0))
    print("抖动基线 %.1f（<= 它 = 锁住）\n" % 抖动基线)
    print("%-26s %-26s %-22s %8s  %s" % ("出图", "背景区平均色", "相对底稿色差", "全图差异", "判定"))
    print("-" * 104)

    面板 = []
    for p in 出图们:
        if not os.path.exists(p):
            print("%-26s 缺文件" % os.path.basename(p))
            continue
        a = 载入(p, 尺寸)
        w1 = 背景均值(a)
        d = w1 - w0
        幅度 = float(np.abs(d).mean())
        全差 = float((差异图(a, 底) > 25).mean() * 100)
        print("%-26s RGB(%5.1f,%5.1f,%5.1f)      %+6.1f %+6.1f %+6.1f (%4.1f)  %6.1f%%  %s"
              % (os.path.basename(p), *w1, *d, 幅度, 全差, 判定(幅度)))
        热 = np.zeros_like(a)
        热[..., 0] = np.clip(差异图(a, 底) * 3, 0, 255)
        面板.append((os.path.basename(p), a, 热))
    if not 面板:
        return 1
    出总览([("底稿", 底, None)] + 面板, 输出目录)
    return 0


def 模式_两两(路径们, 输出目录):
    存在的 = [p for p in 路径们 if os.path.exists(p)]
    if len(存在的) < 2:
        print("!! 至少给两张图（现有 %d 张）" % len(存在的))
        return 1
    首 = Image.open(存在的[0]).convert("RGB")
    告知 = 定配置(main.args, 首)
    for 行 in 告知:
        print(行)

    # 画幅一致性：几张图原始尺寸不一致 = 送进模型的帧不同版/不同画幅，必须报出来
    原始 = {os.path.basename(p): Image.open(p).size for p in 存在的}
    不同 = {v for v in 原始.values()}
    if len(不同) > 1:
        print("\n⚠️  这几张图的**原始尺寸不一致** —— 不同画幅/不同版本的嫌疑，先查清：")
        for n, s in 原始.items():
            print("      %-26s %dx%d" % (n, s[0], s[1]))
    print()

    图 = [(os.path.basename(p), 载入(p, 尺寸)) for p in 存在的]
    print("两两互比（背景区色差 / 全图差异%）")
    print("判据：<= %.1f = 同一版本；>= 10 = 不同版本（背景被重画）\n" % 抖动基线)
    头 = "%-14s" % ""
    for n, _ in 图:
        头 += "%-16s" % n[:14]
    print(头)
    for i, (na, a) in enumerate(图):
        行 = "%-14s" % na[:14]
        for j, (nb, b) in enumerate(图):
            if i == j:
                行 += "%-16s" % "—"
                continue
            幅度 = float(np.abs(背景均值(a) - 背景均值(b)).mean())
            全差 = float((差异图(a, b) > 25).mean() * 100)
            标记 = "✓" if 幅度 <= 抖动基线 else ("~" if 幅度 < 10 else "✗")
            行 += "%-16s" % ("%s %4.1f /%4.1f%%" % (标记, 幅度, 全差))
        print(行)
    print("\n✓=同版本  ~=轻微  ✗=不同版本")
    return 0


def 出总览(面板, 输出目录):
    s = 0.36
    w, h = int(尺寸[0] * s), int(尺寸[1] * s)
    列数 = sum(2 if 热 is not None else 1 for _, _, 热 in 面板)
    画布 = Image.new("RGB", (w * 列数 + 12 * (列数 - 1), h + 34), (250, 250, 250))
    dr = ImageDraw.Draw(画布)
    x = 0
    for 名, a, 热 in 面板:
        画布.paste(Image.fromarray(a.astype(np.uint8)).resize((w, h)), (x, 34))
        dr.text((x + 4, 10), 名[:34], fill=(40, 40, 40))
        x += w + 12
        if 热 is not None:
            画布.paste(Image.fromarray(热.astype(np.uint8)).resize((w, h)), (x, 34))
            dr.text((x + 4, 10), "差异热力图", fill=(150, 60, 160))
            x += w + 12
    os.makedirs(输出目录, exist_ok=True)
    目的地 = os.path.join(输出目录, "_比对_总览.jpg")
    画布.save(目的地, quality=94)
    print("\n-> %s" % 目的地)
    return 目的地


def main():
    ap = argparse.ArgumentParser(description="改图后 / 生视频前的核验关卡")
    ap.add_argument("--底稿", help="模式 A：基准图")
    ap.add_argument("--出图", nargs="+", help="模式 A：待核验的出图")
    ap.add_argument("--两两", nargs="+", help="模式 B：要互比的几帧")
    ap.add_argument("--输出", default="", help="总览图输出目录，默认取底稿/首图所在目录")
    ap.add_argument("--画幅", default="9:16", choices=sorted(画幅预设.keys()),
                    help="背景区预设（16:9 的片子务必写；默认 9:16 是历史口径）")
    ap.add_argument("--尺寸", default="", help="强制统一尺寸，如 1280x720；默认取输入图原始尺寸")
    ap.add_argument("--背景区", default="", help="手动采样框 'x0,y0,x1,y1;...'（比例值）；不给就按 --画幅 取预设")
    ap.add_argument("--抖动基线", type=float, default=None, help="<= 它 = 锁住；逐项目实测后传入（不给默认 3.0）")
    args = ap.parse_args()
    main.args = args          # 定配置 要读它，传参链条太长，挂这里

    if args.底稿 and args.出图:
        目录 = args.输出 or os.path.dirname(os.path.abspath(args.底稿))
        return 模式_对比(args.底稿, args.出图, 目录)
    if args.两两:
        目录 = args.输出 or os.path.dirname(os.path.abspath(args.两两[0]))
        return 模式_两两(args.两两, 目录)
    print("!! 用法：--底稿 X --出图 A B C  或  --两两 A B C")
    return 1


if __name__ == "__main__":
    sys.exit(main())
