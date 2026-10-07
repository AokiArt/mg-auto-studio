#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成片拼接唯一入口 —— 关卡 J（视频验收）。

【为什么有这个文件】
  用户 2026-10-06 定的「视频递归闭环」规则：
    「你要生成的新视频要判断一下它有没有 bug，老的 bug 有没有改掉；没改掉就改提示词重生成；
      有没有引入新问题，引入了新问题也要重新做 —— 这也是一个递归。一直到最后完全通过为止。
      并且你可以只单独生成某一段，但**你要保证它是能够无缝连接进去的**。
      所以你取的帧的位置、还包括那个切法就很重要 …… 就是会有一个多的点就是**切割的位置**，
      然后这个位置要记住，并且是参考要出现。」
  按 [[rules-need-gates-not-docs]]：**写进文档挡不住，要做成跑不起来的关卡**。
  所以这里不提醒你"记得对一下接缝"，而是：**这个文件不点头，成片文件根本不会被写出来。**

【它凭什么能挡住 —— 四查，前两条跟具体毛病无关】
  一·切点闭环   台账里每段必须写出「在源里的起止帧」和「在成片里的绝对入帧」。
                脚本自己算：段 i 的入帧 必须 == 段 i-1 的入帧 + 段长。对不上就爆。
                —— 这就是「切割的位置要记住」的机械化：记不住就拼不出来。
  二·切出来真对 真去切，切完数帧：**切出的帧数必须 == 止帧-起帧+1**，一帧不多一帧不少。
                —— 这一条专门防即梦的坑：它容器把帧率标成 60，实际内容 ~24fps，
                   **按秒切必错**，只能按帧号切。切错了这里立刻爆。
  三·无缝校验   每一道接缝，抽「前段最后一帧」和「后段第一帧」逐像素比。
                台账里每一刀都要自己声明是**硬切**（换场景，本来就该不同）还是**续接**（同场景
                续接，必须看不出跳），两者用**相反的判据**，两头堵：
                  续接 → 腐5核 必须 ≤ 0.20%（接得上）
                  硬切 → 腐5核 必须 ≥ 20%（确实是换场景）—— 低于它 = 拿硬切蒙混接不上的缝
                「腐5核」= 差异掩码腐蚀 5x5 之后还剩多少。用它的理由：整幅图被挪半像素时
                所有边缘都会亮，原始差异会虚高（实测缝2 原始 5.40% 看着吓人，其实两格
                长得一模一样）。腐蚀掉细边，**细边没了=同一张图轻微位移=接得上；
                成片红还在=内容真换了**。实测分离度两三个数量级，中间是空的：
                  视频内相邻帧 0.00 / 真接得上 0.01 / 即梦重绘误差 0.86 / 换场景 88.75
  四·销项与新账 验收单里：老问题逐条必须销掉=true；新问题逐条登记、「待修」不许过；
                每段抽的每一格都要写一句"这一格里有什么"（写不出 = 没看）。
                结论必须是「通过」。
  前三条是硬的，跟"这次是什么毛病"无关；第四条逼你真的看过。

用法：
  # ① 先只看（出接缝图、接片图、台账校验，不写成品）
  python3 tools/验收视频.py --台账 拼接台账_块1.json --看
  # ② 逐张看完、填好验收单，再来拼
  python3 tools/验收视频.py --台账 拼接台账_块1.json --验收单 验收单_块1.json --成品 成片/块1_xxx.mp4
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

帧率默认 = 24
续接阈值默认 = 0.20       # 「续接」缝：腐蚀核占比上限（%）
硬切下限默认 = 20.0       # 「硬切」缝：腐蚀核占比下限（%）—— 低于它就是拿硬切蒙混
必填字 = 10               # 每格「看到」至少这么多字
否定词 = ("无", "没有", "没问题", "正常", "空", "干净", "ok", "OK", "好", "同上", "-", "。")
抽帧数 = 6                # 每段抽几格进接片图（含首末帧）
差异阈 = 25               # 单像素算「变了」的灰度差


def 打印(*a):
    print(*a)
    sys.stdout.flush()


def 跑(命令, 要输出=False):
    r = subprocess.run(命令, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("命令失败：%s\n%s" % (" ".join(命令[:6]), r.stderr.decode()[:500]))
    return r.stdout.decode(errors="replace") if 要输出 else ""


def 探信息(路径):
    """返回 (宽, 高, 帧数, 声称帧率)。注：**帧数才是真话**，帧率字段不可信。"""
    出 = 跑(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
             "-of", "json", 路径], 要输出=True)
    s = (json.loads(出).get("streams") or [{}])[0]
    帧数 = int(s.get("nb_frames") or 0)
    if 帧数 == 0:                       # 有些容器不报 nb_frames，数一遍
        出2 = 跑(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames", "-of", "json", 路径], 要输出=True)
        帧数 = int((json.loads(出2).get("streams") or [{}])[0].get("nb_read_frames") or 0)
    if 帧数 == 0:
        raise RuntimeError("数不出帧数（nb_frames 与 count_frames 都为空）：" + 路径)
    有音 = 跑(["ffprobe", "-v", "error", "-select_streams", "a:0",
              "-show_entries", "stream=index", "-of", "csv=p=0", 路径], 要输出=True).strip() != ""
    return int(s.get("width") or 0), int(s.get("height") or 0), 帧数, s.get("r_frame_rate") or "?", 有音


def 取帧(视频, n, 出):
    """按**帧号**取帧。绝不用 -ss 按秒取 —— 见文件头「切点闭环」。"""
    跑(["ffmpeg", "-v", "error", "-i", 视频, "-vf", "select=eq(n\\,%d)" % n,
        "-frames:v", "1", "-y", 出])
    im = cv2.imread(出)
    if im is None:
        raise RuntimeError("取帧失败：%s 第 %d 帧" % (视频, n))
    return im


def 切段(视频, 起帧, 止帧, 出):
    """按帧号切 [起帧, 止帧]（含两端），重定时为 24fps CFR。切完自己数帧数。"""
    跑(["ffmpeg", "-v", "error", "-i", 视频,
        "-vf", "select=between(n\\,%d\\,%d),setpts=N/%d/TB" % (起帧, 止帧, 帧率默认),
        "-fps_mode", "cfr", "-r", str(帧率默认),
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", 出])
    _, _, 帧数, _, _ = 探信息(出)
    应 = 止帧 - 起帧 + 1
    if 帧数 != 应:
        raise RuntimeError("切段帧数对不上：%s 第 %d..%d 应得 %d 帧，实得 %d 帧"
                           % (os.path.basename(视频), 起帧, 止帧, 应, 帧数))
    return 帧数


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


def 存jpg(路径, im, q=93):
    cv2.imwrite(路径, im, [int(cv2.IMWRITE_JPEG_QUALITY), q])


def 标字(bgr, 文, 色=(255, 255, 255), 倍=0.9, 粗=2, 位=None):
    cv2.putText(bgr, 文, 位 or (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 倍, (0, 0, 0), 粗 + 3, cv2.LINE_AA)
    cv2.putText(bgr, 文, 位 or (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 倍, 色, 粗, cv2.LINE_AA)
    return bgr


def 出接缝图(前帧, 后帧, 名, 出夹, 缝号):
    """A 末帧 | B 首帧 | 差异热力图（红=差异>阈）。

    返回 (粗占比%, 腐5核占比%, 路径)。**两个数的分工：**
      粗占比  —— 只要整幅图被挪了半像素，所有边缘都会亮 → 粗占比虚高（缝2 就吃了这个亏，
                 5.40% 看着吓人，其实两格长得一模一样）。
      腐5核    —— 把细边腐蚀掉之后还剩多少。**细边被吃光 = 同一张图轻微位移 = 接得上；
                 成片红还在 = 内容真换了 = 接不上。** 实测分离度两三个数量级：
                 视频内相邻帧 0.00 / 真接得上 0.01 / 即梦重绘误差 0.86 / 换场景 88.75。
    """
    if 前帧.shape != 后帧.shape:
        后帧 = cv2.resize(后帧, (前帧.shape[1], 前帧.shape[0]), interpolation=cv2.INTER_AREA)
    差 = np.abs(前帧.astype(np.int16) - 后帧.astype(np.int16)).max(2).astype(np.uint8)
    变 = (差 > 差异阈).astype(np.uint8)
    粗 = float(变.mean() * 100)
    核 = float(cv2.erode(变, np.ones((5, 5), np.uint8)).mean() * 100)
    热 = cv2.applyColorMap(np.clip(差.astype(np.int32) * 3, 0, 255).astype(np.uint8),
                           cv2.COLORMAP_INFERNO)
    热[变 > 0] = (0, 0, 255)
    a = 标字(前帧.copy(), "A last", (0, 255, 255))
    b = 标字(后帧.copy(), "B first", (0, 255, 255))
    c = 标字(热, "raw %.2f%%   core %.2f%%" % (粗, 核), (255, 255, 255), 0.7)
    k = min(1.0, 560.0 / a.shape[1])
    缩 = lambda im: cv2.resize(im, (int(im.shape[1] * k), int(im.shape[0] * k)),
                               interpolation=cv2.INTER_AREA)
    a, b, c = 缩(a), 缩(b), 缩(c)
    缝 = np.full((a.shape[0], 10, 3), 255, np.uint8)
    拼 = np.hstack([a, 缝, b, 缝, c])
    路 = os.path.join(出夹, "接缝%d_%s.jpg" % (缝号, 名.replace("/", "_")))
    存jpg(路, 拼)
    return 粗, 核, 路


def 出接片(帧们, 名, 出夹, 列=6, 格宽=300):
    """帧们 = [(帧号, 绝对帧号, BGR)]；排成一屏看完。"""
    from PIL import Image, ImageDraw
    if not 帧们:
        return None
    行 = (len(帧们) + 列 - 1) // 列
    比 = 帧们[0][2].shape[0] / float(帧们[0][2].shape[1])
    格高 = int(round(格宽 * 比))
    边, 标注, 头高 = 8, 26, 46
    宽 = 边 + 列 * (格宽 + 边)
    高 = 头高 + 行 * (格高 + 标注 + 边)
    画 = Image.new("RGB", (宽, 高), (26, 26, 26))
    d = ImageDraw.Draw(画)
    d.text((10, 6), "段「%s」逐格接片 —— 每一格都要在验收单里写「看到」" % 名,
           font=取字体(24), fill=(120, 220, 255))
    d.text((10, 28), "格子名 = 段名@绝对帧号；从左到右是时间顺序",
           font=取字体(18), fill=(170, 170, 170))
    字 = 取字体(20)
    for i, (帧号, 绝帧, s) in enumerate(帧们):
        r, c = divmod(i, 列)
        x = 边 + c * (格宽 + 边)
        y = 头高 + r * (格高 + 标注 + 边)
        图 = Image.fromarray(cv2.cvtColor(cv2.resize(s, (格宽, 格高),
                                                    interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2RGB))
        画.paste(图, (x, y))
        d.rectangle([x - 1, y - 1, x + 格宽, y + 格高], outline=(70, 70, 70))
        d.text((x + 2, y + 格高 + 3), "帧%d   %s@%d" % (帧号, 名, 绝帧), font=字, fill=(235, 235, 235))
    路 = os.path.join(出夹, "接片_%s.jpg" % 名.replace("/", "_"))
    画.save(路, quality=92)
    return 路


def 找源(素材根, 台账夹, 源路):
    """台账里的「源」按**项目根**相对写；找不到再退回台账所在目录试试。"""
    if os.path.isabs(源路):
        return 源路 if os.path.exists(源路) else None
    for 基 in (素材根, 台账夹):
        if not 基:
            continue
        p = os.path.abspath(os.path.join(基, 源路))
        if os.path.exists(p):
            return p
    return None


def 读台账(路径, 台账夹, 素材根):
    if not os.path.exists(路径):
        raise SystemExit("!! 台账不存在：%s" % 路径)
    with open(路径, encoding="utf-8") as f:
        账 = json.load(f)
    段 = 账.get("段落") or []
    if not 段:
        raise SystemExit("!! 台账里一条「段落」都没有 —— 没写切点，就拼不出无缝的片子。")
    问 = []
    for i, s in enumerate(段, 1):
        for k in ("名", "源", "起帧", "止帧", "入帧", "说明"):
            if s.get(k) in (None, ""):
                问.append("第 %d 段没写「%s」" % (i, k))
        if len(str(s.get("说明") or "").strip()) < 必填字:
            问.append("第 %d 段「%s」的说明只有 %d 字（要求 ≥%d）—— 写不出为什么从这里切 = 没想清楚"
                      % (i, s.get("名"), len(str(s.get("说明") or "").strip()), 必填字))
        p = 找源(素材根, 台账夹, str(s.get("源") or ""))
        if p is None:
            问.append("第 %d 段「%s」的源文件不存在（按项目根 %s 找过）：%s"
                      % (i, s.get("名"), 素材根, s.get("源")))
        else:
            s["_源"] = p
        if None not in (s.get("起帧"), s.get("止帧")) and s["起帧"] > s["止帧"]:
            问.append("第 %d 段「%s」起帧 %s > 止帧 %s" % (i, s.get("名"), s["起帧"], s["止帧"]))
        # —— 前缝：跟上一段之间这一刀，是「硬切」还是「续接」？必须自己说清 ——
        if i == 1:
            if s.get("前缝"):
                问.append("第 1 段不该有「前缝」（它前面没有东西）")
        else:
            fq = s.get("前缝") or {}
            if fq.get("类型") not in ("硬切", "续接"):
                问.append("第 %d 段「%s」的「前缝.类型」必须是 硬切/续接，现在是 %r"
                          % (i, s.get("名"), fq.get("类型")))
            elif len(str(fq.get("理由") or "").strip()) < 必填字:
                问.append("第 %d 段「%s」的「前缝.理由」只有 %d 字（要求 ≥%d）—— 凭什么这一刀是「%s」"
                          % (i, s.get("名"), len(str(fq.get("理由") or "").strip()), 必填字, fq.get("类型")))
    if 问:
        raise SystemExit("🔴 台账不合格 —— 一个字节都没拼：\n   · " + "\n   · ".join(问))
    return 账, 段


def 主():
    ap = argparse.ArgumentParser(description="成片拼接唯一入口：切点闭环 + 切出真对 + 无缝校验 + 逐条销项")
    ap.add_argument("--台账", required=True, help="拼接台账 JSON（切点都在这）")
    ap.add_argument("--验收单", default="", help="逐段逐格验收 JSON；不给就只出材料")
    ap.add_argument("--成品", default="", help="成片写到这里；不给就只检查不落盘")
    ap.add_argument("--看", action="store_true", help="只出材料（接缝图/接片图），然后停")
    ap.add_argument("--覆盖", action="store_true", help="允许覆盖已存在的成品（默认拒绝）")
    ap.add_argument("--出夹", default="")
    ap.add_argument("--项目根", default="", help="台账里「源」按哪个目录相对找，默认当前目录")
    args = ap.parse_args()

    根 = os.path.dirname(os.path.abspath(args.台账))
    素材根 = os.path.abspath(args.项目根 or os.getcwd())
    账, 段 = 读台账(args.台账, 根, 素材根)
    帧率 = int(账.get("帧率") or 帧率默认)
    续阈 = float(账.get("续接阈值", 续接阈值默认))
    硬限 = float(账.get("硬切下限", 硬切下限默认))
    出夹 = args.出夹 or os.path.join(根, "验收自检",
                                   "%s_%s" % (账.get("块", "片"), datetime.datetime.now().strftime("%m%d_%H%M%S")))
    os.makedirs(出夹, exist_ok=True)

    打印("=" * 70)
    打印("【关卡 J · 视频验收】%s" % 账.get("块", "（未命名）"))
    打印("=" * 70)

    # ══ 查一 · 切点闭环 ══
    期 = 0
    参 = []
    for i, s in enumerate(段, 1):
        源 = s["_源"]
        w, h, 总, 称帧率, 有音 = 探信息(源)
        长 = s["止帧"] - s["起帧"] + 1
        ok入 = (s["入帧"] == 期)
        ok界 = (0 <= s["起帧"] <= s["止帧"] < 总)
        打印("  段%d「%s」" % (i, s["名"]))
        打印("     源 %s  %dx%d  %d帧  声称%s  音轨%s"
              % (os.path.relpath(源, 素材根), w, h, 总, 称帧率, "有" if 有音 else "无"))
        打印("     切点 源帧 %d..%d（%d帧）  →  成片绝对帧 %d..%d  %s"
              % (s["起帧"], s["止帧"], 长, 期, 期 + 长 - 1, "✅" if ok入 else "🔴 对不上上一段的尾！"))
        打印("     说明 %s" % s["说明"])
        if not ok入:
            raise SystemExit("🔴 切点不闭环：第 %d 段声明入帧 %d，但上一段拼到 %d。"
                             "（切割位置必须首尾相接，不许有缝、不许重叠）" % (i, s["入帧"], 期))
        if not ok界:
            raise SystemExit("🔴 第 %d 段源帧号 %d..%d 超出源文件 [0..%d]"
                             % (i, s["起帧"], s["止帧"], 总 - 1))
        参.append((i, s, 源, 长))
        期 += 长
    总帧 = 期
    打印("")
    打印("  ① 切点闭环 ✅   %d 段拼成 %d 帧 = %.3f 秒 @%dfps"
          % (len(段), 总帧, 总帧 / float(帧率), 帧率))
    if 账.get("总帧数") not in (None, "") and int(账["总帧数"]) != 总帧:
        raise SystemExit("🔴 台账写的总帧数 %s ≠ 实际拼出 %d 帧" % (账["总帧数"], 总帧))
    打印("     ⚠ 即梦输出的容器帧率字段不可信（常报 60/1），实际内容 ~%dfps —— "
         "所以本脚本**只按帧号切**，绝不按秒切。" % 帧率)

    # ══ 查二 · 切出来真对（真切一遍，数帧）══
    工作 = os.path.join(出夹, "切段")
    os.makedirs(工作, exist_ok=True)
    切文件 = []
    for i, s, 源, 长 in 参:
        出 = os.path.join(工作, "seg%02d.mp4" % i)
        实 = 切段(源, s["起帧"], s["止帧"], 出)
        print("  ② 第 %d 段真切：应 %d 帧，实 %d 帧  ✅" % (i, 长, 实))
        切文件.append(出)

    # ══ 查三 · 无缝校验（按缝的类型分开判）══
    打印("")
    打印("  ③ 无缝校验   续接缝：核 ≤ %.2f%% 才算接得上   硬切缝：核 ≥ %.2f%% 才算是真换场景"
         % (续阈, 硬限))
    缝 = []
    坏缝 = []
    for k in range(len(参) - 1):
        i, s, 源, _ = 参[k]
        j, s2, 源2, _ = 参[k + 1]
        类 = ((s2.get("前缝") or {}).get("类型") or "?")
        前 = 取帧(源, s["止帧"], os.path.join(出夹, "_缝A.png"))
        后 = 取帧(源2, s2["起帧"], os.path.join(出夹, "_缝B.png"))
        粗, 核, 路 = 出接缝图(前, 后, "%s-%s" % (s["名"], s2["名"]), 出夹, k + 1)
        if 类 == "续接":
            好 = 核 <= 续阈
            判 = "✅ 接得上" if 好 else "🔴 接不上"
            边 = "上限 %.2f%%" % 续阈
        else:
            好 = 核 >= 硬限
            判 = "✅ 确实是硬切" if 好 else "🔴 这不是硬切，别拿硬切蒙混"
            边 = "下限 %.2f%%" % 硬限
        print("     缝%d [%s]  %s(源#%d) ⇄ %s(源#%d)" % (k + 1, 类, s["名"], s["止帧"], s2["名"], s2["起帧"]))
        print("           粗差异 %6.2f%%   腐5核 %6.2f%%（%s）   %s" % (粗, 核, 边, 判))
        缝.append((k + 1, "%s ⇄ %s" % (s["名"], s2["名"]), 类, 粗, 核, 好, 路))
        if not 好:
            坏缝.append(k + 1)

    # ══ 出接片图：每段抽 抽帧数 格 ══
    打印("")
    格 = []
    for i, s, 源, 长 in 参:
        序 = sorted({int(round(s["起帧"] + t * (长 - 1) / float(抽帧数 - 1)))
                     for t in range(抽帧数)})
        帧们 = []
        for n in 序:
            im = 取帧(源, n, os.path.join(出夹, "_抽.png"))
            k = min(1.0, 300.0 / im.shape[1])
            小 = cv2.resize(im, (300, int(round(im.shape[0] * k))), interpolation=cv2.INTER_AREA)
            帧们.append((n, s["入帧"] + (n - s["起帧"]), 小))
            格.append("%s@%d" % (s["名"], s["入帧"] + (n - s["起帧"])))
        路 = 出接片(帧们, s["名"], 出夹)
        打印("  段%d 接片：%s（%d 格）" % (i, os.path.relpath(路, 根), len(帧们)))
    for f in ("_缝A.png", "_缝B.png", "_抽.png"):
        p = os.path.join(出夹, f)
        if os.path.exists(p):
            os.remove(p)

    打印("")
    打印("  材料：%s" % os.path.relpath(出夹, 根))
    打印("    接缝1..%d.jpg  +  接片_<段名>.jpg" % max(0, len(参) - 1))
    硬伤 = bool(坏缝)

    if args.看 or not args.验收单:
        打印("")
        打印("👉 下一步：把每张 接缝*.jpg 和 接片_*.jpg **都看一遍**，")
        打印("   然后写验收单：老问题逐条销项、新问题逐条登记、每格写一句「看到」，")
        打印("   再带 --验收单 跑一次。没看就别写；写了就是假账。")
        return 0

    # ══ 查四 · 销项与新账 ══
    单路径 = os.path.abspath(args.验收单)
    if not os.path.exists(单路径):
        raise SystemExit("!! 验收单不存在：%s" % 单路径)
    with open(单路径, encoding="utf-8") as f:
        单 = json.load(f)

    问 = []
    看过 = {b.get("格"): b for b in (单.get("逐格") or [])}
    for g in 格:
        b = 看过.get(g)
        if not b:
            问.append("接片格 %s 没写 —— 没看就别拼" % g)
            continue
        看 = str(b.get("看到") or "").strip()
        if len(看) < 必填字:
            问.append("格 %s 的「看到」只有 %d 字（要求 ≥%d）" % (g, len(看), 必填字))
        elif 看 in 否定词:
            问.append("格 %s 的「看到」是空话（%s）" % (g, 看))
        if b.get("判定") not in ("无问题", "有问题"):
            问.append("格 %s 的「判定」必须是 无问题/有问题" % g)

    缝看 = {s.get("缝"): s for s in (单.get("接缝") or [])}
    for 号, 名, 类, 粗, 核, 好, _ in 缝:
        s = 缝看.get(号)
        if not s:
            问.append("接缝 %d（%s）没写 —— 接缝是必须亲自看的一格" % (号, 名))
            continue
        if len(str(s.get("看到") or "").strip()) < 必填字:
            问.append("接缝 %d 的「看到」只有 %d 字（要求 ≥%d）—— 这是硬切还是接得上，一眼就能说清"
                      % (号, len(str(s.get("看到") or "").strip()), 必填字))
        if s.get("判定") not in ("对", "不对"):
            问.append("接缝 %d 的「判定」必须是 对/不对（现在 %r）" % (号, s.get("判定")))
        elif s.get("判定") == "不对":
            问.append("接缝 %d 你自己判了「不对」—— 不对就不许拼" % 号)

    老 = 单.get("老问题销项") or []
    清单老 = 账.get("老问题") or []
    if 清单老:
        销 = {s.get("号"): s for s in 老}
        for 条 in 清单老:
            号 = 条.get("号")
            s = 销.get(号)
            if not s:
                问.append("老问题第 %s 号（%s）没销项判定" % (号, 条.get("描述")))
                continue
            if s.get("销掉") is not True:
                问.append("老问题第 %s 号还判「没销掉」—— 有没销的就不许拼" % 号)

    新 = 单.get("新问题") or []
    for i, t in enumerate(新, 1):
        for k in ("描述", "位置", "处置"):
            if not str(t.get(k) or "").strip():
                问.append("新问题第 %d 条没写「%s」" % (i, k))
        if t.get("处置") == "待修":
            问.append("新问题第 %d 条还挂「待修」—— 待修就不许拼" % i)
    有问题的格 = [b.get("格") for b in (单.get("逐格") or []) if b.get("判定") == "有问题"]
    if 有问题的格 and not 新:
        问.append("这些格判了「有问题」却一条新问题都没登记：%s" % "、".join(有问题的格))
    if not 新:
        因 = str(单.get("新问题为空的原因") or "").strip()
        if len(因) < 必填字:
            问.append("新问题为空，必须写清「新问题为空的原因」（≥%d 字）—— 凭什么说没有新问题" % 必填字)
    if 单.get("结论") != "通过":
        问.append("验收单「结论」不是 通过（现在是 %r）" % 单.get("结论"))

    if 问:
        打印("")
        打印("🔴 验收单没过 —— **成片一个字节都没写**：")
        for p in 问:
            打印("   · %s" % p)
        return 1
    if 硬伤:
        打印("")
        打印("🔴 无缝校验没过（缝 %s）—— 验收单填了也不算，成片不写。"
              % "、".join(str(x) for x in 坏缝))
        打印("   续接缝没过 → 换切点（挪到动得最少的那帧）或重生成那一段；"
              "硬切缝没过 → 你标错类型了，它压根不是场景切换。")
        return 1

    if not args.成品:
        打印("\n✅ 四查全过。（没给 --成品，所以只报告不落盘）")
        return 0

    成品 = os.path.abspath(args.成品)
    if os.path.exists(成品) and not args.覆盖:
        raise SystemExit("!! 成品已存在，不覆盖：%s\n   换个文件名，或显式加 --覆盖。" % os.path.relpath(成品, 根))
    if os.path.dirname(成品):
        os.makedirs(os.path.dirname(成品), exist_ok=True)
    清 = os.path.join(工作, "list.txt")
    with open(清, "w", encoding="utf-8") as f:
        for p in 切文件:
            f.write("file '%s'\n" % os.path.abspath(p).replace("'", "'\\''"))
    跑(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", 清,
        "-c", "copy", "-movflags", "+faststart", "-y", 成品])
    _, _, 出帧, _, 出音 = 探信息(成品)
    if 出帧 != 总帧:
        打印("🔴 拼出来 %d 帧，应该是 %d 帧 —— 成片已写出但有毛病，别用。" % (出帧, 总帧))
        return 1

    报告 = os.path.join(出夹, "验收报告.md")
    with open(报告, "w", encoding="utf-8") as f:
        f.write("# 视频验收报告 · %s\n\n" % 账.get("块", ""))
        f.write("- 台账：%s\n- 成品：%s\n- 时间：%s\n\n"
                % (os.path.relpath(args.台账, 根), os.path.relpath(成品, 根),
                   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        f.write("## 切点闭环\n\n| 段 | 源 | 源帧 | 成片绝对帧 | 帧数 |\n|---|---|---|---|---|\n")
        for i, s, 源, 长 in 参:
            f.write("| %d %s | %s | %d..%d | %d..%d | %d |\n"
                    % (i, s["名"], s["源"], s["起帧"], s["止帧"], s["入帧"], s["入帧"] + 长 - 1, 长))
        f.write("\n**合计 %d 帧 = %.3f 秒 @%dfps**\n\n" % (总帧, 总帧 / float(帧率), 帧率))
        f.write("## 无缝校验\n\n"
                "判据用**腐蚀核**而不是原始差异：整幅图被挪半像素时所有边缘都会亮，原始差异会虚高；"
                "把细边腐蚀掉之后，**细边没了=同一张图轻微位移=接得上；成片红还在=内容真换了**。\n\n"
                "| 缝 | 接缝 | 类型 | 粗差异 | 腐5核 | 判据 | 判定 |\n|---|---|---|---|---|---|---|\n")
        for 号, 名, 类, 粗, 核, 好, _ in 缝:
            边 = ("≤%.2f%%" % 续阈) if 类 == "续接" else ("≥%.2f%%" % 硬限)
            f.write("| %d | %s | %s | %.2f%% | %.2f%% | %s | %s |\n"
                    % (号, 名, 类, 粗, 核, 边, "过" if 好 else "**没过**"))
        f.write("\n## 逐格\n\n| 格 | 看到 | 判定 |\n|---|---|---|\n")
        for b in 单.get("逐格") or []:
            f.write("| %s | %s | %s |\n" % (b.get("格"), b.get("看到"), b.get("判定")))
        f.write("\n## 新问题\n\n")
        if 新:
            for t in 新:
                f.write("- 【%s】%s @ %s\n" % (t.get("处置"), t.get("描述"), t.get("位置")))
        else:
            f.write("- 无。原因：%s\n" % 单.get("新问题为空的原因"))
        f.write("\n## 老问题销项\n\n| 号 | 销掉 | 说明 |\n|---|---|---|\n")
        for s in 老:
            f.write("| %s | %s | %s |\n" % (s.get("号"), s.get("销掉"), s.get("说明", "")))
        f.write("\n**结论：通过**\n")
    打印("")
    打印("✅ 四查全过。成片 → %s（%d 帧 / %.3f 秒，%s）"
          % (os.path.relpath(成品, 根), 出帧, 出帧 / float(帧率),
             "带音轨" if 出音 else "无音轨"))
    打印("   验收报告 → %s" % os.path.relpath(报告, 根))
    if 出音:
        打印("   ⚠ 本脚本切段一律 -an，成片理应无音 —— 它却带了音轨，去查 ffmpeg 有没有漏。")
    return 0


if __name__ == "__main__":
    sys.exit(主())
