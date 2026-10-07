#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成品唯一入口 —— 案例 2026-10-06 事件之后加的关卡 I（成图自检）。

【为什么有这个文件】
  规矩（"改完要查有没有引入新问题"）在 SKILL.md 里写了，照样没执行：v5 交出去时
  版图边被矩形切出两条缝、头顶的树被切一半、老人物的旗还留着 —— 一个都没查。
  按 [[rules-need-gates-not-docs]]：**写进文档挡不住，要做成跑不起来的关卡**。
  所以这里不提醒你"记得看图"，而是：**这个文件不点头，成品文件根本不会被写出来。**

【它凭什么能抓出上面那三个毛病 —— 两查是通用的，跟具体毛病无关】
  你得先声明「这一版允许变的那块」，做成一张掩码（白=允许变，黑=不许动）。
  然后：
    越界 = 变了但不在掩码里   → 抓「把不该动的地方切了一块」（v5 的版图下沿那两条缝）
    漏改 = 在掩码里但没变     → 抓「该换的没换到」（v5 被切掉的树顶、没盖住的老像素）
  这两个检查对任何一张图都成立，不依赖"这次是什么毛病"。声明得松（想蒙混）→ 漏改立刻爆；
  声明得紧 → 越界立刻爆。两头堵，掩码只能是真的。

【另外强制你"看"】
  自动检查只能抓几何上的错，抓不了"这张脸画丑了"。所以收口前必须先出一套**分块放大图**，
  并且自检单里**每一块都要写一句"这一格里有什么"**（写不出 = 没看）。
  再把「帧问题清单」逐条销项、把新问题逐条登记。

用法：
  # ① 先只看（生成必须看的材料，不写成品）
  python3 tools/收口.py --清单 镜头清单.json --镜头 02 \
      --底稿 中间产物_参考帧_v4.png --候选 _候选_v6.png --改区 v6_允许变区.png --看
  # ② 逐块看完、填好自检单，再来收口
  python3 tools/收口.py ... --自检单 自检单_02_v6.json --成品 中间产物_参考帧_v6.png
"""
import argparse
import datetime
import json
import os
import shutil
import sys

import cv2
import numpy as np

分块默认 = "3x4"          # 列 x 行
越界容差 = 0
漏改容差 = 0
自检单必填块字 = 10       # 每块「看到」至少这么多字
否定词 = ("无", "没有", "没问题", "正常", "空", "干净", "ok", "OK", "好", "同上", "-", "。")


def 打印(*a):
    print(*a)
    sys.stdout.flush()


def 读图(路径, 名):
    if not os.path.exists(路径):
        raise SystemExit("!! %s 不存在：%s" % (名, 路径))
    im = cv2.imread(路径)
    if im is None:
        raise SystemExit("!! %s 读不出来：%s" % (名, 路径))
    return im


def 对齐(im, 形):
    if im.shape[:2] != 形[:2]:
        打印("   （%s 尺寸 %s ≠ %s，自动缩放）" % ("图", im.shape[:2], 形[:2]))
        im = cv2.resize(im, (形[1], 形[0]), interpolation=cv2.INTER_AREA)
    return im


def 掩码化(im):
    """掩码图 → 二值 0/1（灰度高 = 允许变）。"""
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return (im > 127).astype(np.uint8)


def 标块(im, 文, 色=(0, 0, 255)):
    cv2.putText(im, 文, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 色, 2, cv2.LINE_AA)
    return im


def 存jpg(路径, im, q=93):
    cv2.imwrite(路径, im, [int(cv2.IMWRITE_JPEG_QUALITY), q])


def 分块表(形, 分块):
    c, r = [int(v) for v in 分块.lower().split("x")]
    H, W = 形[:2]
    表 = []
    for i in range(r):
        for j in range(c):
            x0, x1 = int(W * j / c), int(W * (j + 1) / c)
            y0, y1 = int(H * i / r), int(H * (i + 1) / r)
            表.append(("r%dc%d" % (i + 1, j + 1), x0, y0, x1, y1))
    return c, r, 表


def 出材料(底, 候, 改区, 变, 越界, 漏改, 出夹, 分块):
    """生成总览 + 分块放大图。返回分块列表。"""
    H, W = 底.shape[:2]
    热 = np.clip(np.abs(候.astype(int) - 底.astype(int)).max(2) * 3, 0, 255).astype(np.uint8)
    热 = cv2.applyColorMap(热, cv2.COLORMAP_INFERNO)

    叠加 = 候.copy()
    叠加[越界 > 0] = (0, 0, 255)          # 红 = 越界（改了不该改的）
    叠加[漏改 > 0] = (255, 0, 0)          # 蓝 = 漏改（该改的没改）
    cs, _ = cv2.findContours(改区, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(叠加, cs, -1, (0, 255, 0), 2)

    缩 = lambda im, k=0.5: cv2.resize(im, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
    行 = np.hstack([缩(标块(底.copy(), "base", (0, 0, 255))),
                    缩(标块(候.copy(), "cand", (0, 0, 255))),
                    缩(热), 缩(标块(cv2.cvtColor(改区 * 255, cv2.COLOR_GRAY2BGR), "allow-mask", (0, 0, 255))),
                    缩(标块(叠加, "red=overreach  blue=missed", (0, 0, 255)))])
    存jpg(os.path.join(出夹, "自检_总览.jpg"), 行)

    c, r, 表 = 分块表(底.shape, 分块)
    夹 = os.path.join(出夹, "自检_分块")
    os.makedirs(夹, exist_ok=True)
    for 名, x0, y0, x1, y1 in 表:
        块 = 叠加[y0:y1, x0:x1].copy()
        cv2.rectangle(块, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), (0, 255, 255), 2)
        标块(块, "%s  x%d-%d y%d-%d" % (名, x0, x1, y0, y1))
        长 = max(x1 - x0, y1 - y0)
        k = min(1.5, 900.0 / 长)
        块 = cv2.resize(块, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_NEAREST)
        存jpg(os.path.join(夹, "%s.jpg" % 名), 块)
    return 表


def 查自检单(单路径, 表, 条目):
    if not os.path.exists(单路径):
        return ["自检单不存在：%s" % 单路径], None
    try:
        with open(单路径, encoding="utf-8") as f:
            单 = json.load(f)
    except Exception as exc:
        return ["自检单不是合法 JSON：%s" % exc], None

    问 = []
    块 = {b.get("格"): b for b in (单.get("分块") or [])}
    for 名, _, _, _, _ in 表:
        b = 块.get(名)
        if not b:
            问.append("分块 %s 没写 —— 没看就别收口" % 名)
            continue
        看 = str(b.get("看到") or "").strip()
        if len(看) < 自检单必填块字:
            问.append("分块 %s 的「看到」只有 %d 字（要求 ≥%d）—— 写不出这一格里有什么 = 没看" % (名, len(看), 自检单必填块字))
        elif 看 in 否定词:
            问.append("分块 %s 的「看到」是空话（%s）—— 要写清这一格里到底有什么" % (名, 看))
        判 = b.get("判定")
        if 判 not in ("无问题", "有问题"):
            问.append("分块 %s 的「判定」必须是 无问题/有问题，现在是 %r" % (名, 判))

    新 = 单.get("新问题") or []
    for i, t in enumerate(新, 1):
        for k in ("描述", "位置", "处置"):
            if not str(t.get(k) or "").strip():
                问.append("新问题第 %d 条没写「%s」" % (i, k))
        if t.get("处置") == "待修":
            问.append("新问题第 %d 条还挂「待修」—— 待修就不许收口" % i)
    有问题的块 = [b.get("格") for b in (单.get("分块") or []) if b.get("判定") == "有问题"]
    if 有问题的块 and not 新:
        问.append("这些块判了「有问题」却一条新问题都没登记：%s" % "、".join(有问题的块))
    if not 新:
        因 = str(单.get("新问题为空的原因") or "").strip()
        if len(因) < 自检单必填块字:
            问.append("新问题为空，必须写清「新问题为空的原因」（≥%d 字）—— 凭什么说没有新问题" % 自检单必填块字)

    销 = {s.get("号"): s for s in (单.get("老清单销项") or [])}
    for 条 in 条目:
        号 = 条.get("号")
        s = 销.get(号)
        if not s:
            问.append("帧问题清单第 %s 号没销项判定" % 号)
            continue
        if s.get("销掉") is not True:
            问.append("帧问题清单第 %s 号还判「没销掉」—— 有没销的就不许定稿" % 号)

    if 单.get("结论") != "通过":
        问.append("自检单「结论」不是 通过（现在是 %r）" % 单.get("结论"))
    return 问, 单


def main():
    ap = argparse.ArgumentParser(description="成品唯一入口：通用两查 + 强制逐块看图 + 逐条销项，全过才写成品")
    ap.add_argument("--清单", required=True)
    ap.add_argument("--镜头", required=True)
    ap.add_argument("--底稿", required=True, help="这一版的底图")
    ap.add_argument("--候选", required=True, help="本地合成出来的待收口图")
    ap.add_argument("--改区", required=True, help="声明「本版允许变」的掩码图（白=允许变）")
    ap.add_argument("--来源", default="", help="这一版用的 AI 原图（只写进报告）")
    ap.add_argument("--成品", default="", help="成品写到这里；不给就只做检查不落盘")
    ap.add_argument("--自检单", default="", help="逐块自检 JSON；不给就只出材料")
    ap.add_argument("--分块", default=分块默认)
    ap.add_argument("--看", action="store_true", help="只生成必须看的材料，然后停")
    ap.add_argument("--覆盖", action="store_true", help="允许覆盖已存在的成品（默认拒绝）")
    ap.add_argument("--出夹", default="")
    args = ap.parse_args()

    根 = os.path.dirname(os.path.abspath(args.清单))
    底 = 读图(args.底稿, "底稿")
    候 = 对齐(读图(args.候选, "候选"), 底.shape)
    改 = 读图(args.改区, "改区")
    if 改.ndim == 3:
        改 = cv2.cvtColor(改, cv2.COLOR_BGR2GRAY)
    if 改.shape[:2] != 底.shape[:2]:
        改 = cv2.resize(改, (底.shape[1], 底.shape[0]), interpolation=cv2.INTER_NEAREST)
    改 = (改 > 127).astype(np.uint8)

    if 改.sum() == 0:
        raise SystemExit("!! 改区掩码全黑 —— 声明「这一版允许变零个像素」？先做掩码。")

    差 = np.abs(候.astype(int) - 底.astype(int)).max(2)
    变 = (差 > 6).astype(np.uint8)
    变 = cv2.morphologyEx(变, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    变 = cv2.morphologyEx(变, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 漏改用**敏感**阈值，越界用**保守**阈值。
    #   两个问题不一样：
    #     越界 = 「我有没有碰到不该碰的地方」→ 只有看得出的改变才算碰（差>6）
    #     漏改 = 「我声明的地方我到底碰没碰」→ 动一下就算碰（差>1）
    #   早先两处共用差>6，结果在**缓慢渐变**的地方假阳性：inpaint 抹过一片近乎同色的
    #   木桌，每个像素都改了 4~5 个灰阶 —— 每一块「零差异像素都是 0」，确实动过，
    #   却因为没跨过 6 被报成「声明了允许变、实际没变」（实测 1641px）。
    #   分开之后，漏改这条反而更严：真正一片没动的地方（差恒为 0）照样爆。
    变_细 = (差 > 1).astype(np.uint8)
    变_细 = cv2.morphologyEx(变_细, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    变_细 = cv2.morphologyEx(变_细, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # —— 通用两查：先各自腐蚀 2px 抹掉抗锯齿边，比的是「核」 ——
    核_变 = cv2.erode(变, np.ones((5, 5), np.uint8))
    核_区 = cv2.erode(改, np.ones((5, 5), np.uint8))
    越界 = (核_变 & ~cv2.dilate(改, np.ones((5, 5), np.uint8))).astype(np.uint8)
    漏改 = (核_区 & ~cv2.dilate(变_细, np.ones((5, 5), np.uint8))).astype(np.uint8)

    出夹 = args.出夹 or os.path.join(根, "收口自检", "%s_%s" % (args.镜头, datetime.datetime.now().strftime("%m%d_%H%M%S")))
    os.makedirs(出夹, exist_ok=True)
    表 = 出材料(底, 候, 改, 变, 越界, 漏改, 出夹, args.分块)

    打印("=" * 68)
    打印("【关卡 I · 成图自检】镜头 %s" % args.镜头)
    打印("=" * 68)
    打印("  改区：%d px（占全图 %.2f%%）" % (int(改.sum()), 改.mean() * 100))
    打印("  变了的：%d px（占全图 %.2f%%）" % (int(变.sum()), 变.mean() * 100))
    打印("  ① 越界（改了不该改的）= %d px  %s" % (int(越界.sum()), "✅" if 越界.sum() <= 越界容差 else "🔴 不合规"))
    打印("  ② 漏改（该改的没改）  = %d px  %s" % (int(漏改.sum()), "✅" if 漏改.sum() <= 漏改容差 else "🔴 不合规"))
    for 名, m in (("越界", 越界), ("漏改", 漏改)):
        if m.sum() == 0:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        序 = np.argsort(-st[1:, 4]) + 1
        打印("     %s 最大几块：%s" % (名, [(int(st[i, 4]),
              "x%d-%d y%d-%d" % (st[i, 0], st[i, 0] + st[i, 2], st[i, 1], st[i, 1] + st[i, 3]))
             for i in 序[:5]]))
    打印("  材料：%s" % os.path.relpath(出夹, 根))
    打印("    自检_总览.jpg  +  自检_分块/ 共 %d 格" % len(表))

    硬伤 = (越界.sum() > 越界容差) or (漏改.sum() > 漏改容差)

    if args.看 or not args.自检单:
        打印("")
        打印("👉 下一步：把 自检_总览.jpg 和 自检_分块/ 里每一格**都看一遍**，")
        打印("   然后照着分块名写自检单（每格一句「看到」+ 判定），再带 --自检单 跑一次。")
        打印("   没看就别写；写了就是假账，最后害的是自己。")
        return 0

    清单路径 = os.path.abspath(args.清单)
    with open(清单路径, encoding="utf-8") as f:
        清单 = json.load(f)
    条目 = ((清单.get("镜头") or {}).get(args.镜头) or {}).get("帧问题清单", {}).get("问题") or []

    问, 单 = 查自检单(args.自检单, 表, 条目)
    if 问:
        打印("")
        打印("🔴 自检单没过 —— **成品一个字节都没写**：")
        for p in 问:
            打印("   · %s" % p)
        return 1
    if 硬伤:
        打印("")
        打印("🔴 通用两查没过（越界 %d / 漏改 %d）—— 自检单填了也不算，成品不写。"
              % (int(越界.sum()), int(漏改.sum())))
        打印("   先把多出来的那块收回去 / 把漏掉的补上，重出候选再来。")
        return 1

    if not args.成品:
        打印("\n✅ 两查过、自检单过。（没给 --成品，所以只报告不落盘）")
        return 0

    成品 = os.path.abspath(args.成品)
    if os.path.exists(成品) and not args.覆盖:
        raise SystemExit("!! 成品已存在，不覆盖：%s\n   要么换个文件名（v7/v8…），要么显式加 --覆盖。"
                         % os.path.relpath(成品, 根))
    if os.path.dirname(成品):
        os.makedirs(os.path.dirname(成品), exist_ok=True)
    shutil.copy2(os.path.abspath(args.候选), 成品)

    报告 = os.path.join(出夹, "自检报告.md")
    with open(报告, "w", encoding="utf-8") as f:
        f.write("# 成图自检报告 · 镜头 %s\n\n" % args.镜头)
        f.write("- 底稿：%s\n- 候选：%s\n- 成品：%s\n" % (args.底稿, args.候选, os.path.relpath(成品, 根)))
        if args.来源:
            f.write("- AI 原图：%s\n" % args.来源)
        f.write("- 时间：%s\n\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("## 通用两查（跟具体毛病无关，任何图都成立）\n\n")
        f.write("| 查什么 | 结果 | 阈值 |\n|---|---|---|\n")
        f.write("| 越界（改了不该改的） | %d px | %d |\n" % (int(越界.sum()), 越界容差))
        f.write("| 漏改（该改的没改） | %d px | %d |\n" % (int(漏改.sum()), 漏改容差))
        f.write("\n## 逐块自检\n\n| 格 | 看到 | 判定 |\n|---|---|---|\n")
        for b in 单.get("分块") or []:
            f.write("| %s | %s | %s |\n" % (b.get("格"), b.get("看到"), b.get("判定")))
        f.write("\n## 新问题\n\n")
        if 单.get("新问题"):
            for t in 单["新问题"]:
                f.write("- 【%s】%s @ %s\n" % (t.get("处置"), t.get("描述"), t.get("位置")))
        else:
            f.write("- 无。原因：%s\n" % 单.get("新问题为空的原因"))
        f.write("\n## 帧问题清单销项\n\n| 号 | 销掉 | 说明 |\n|---|---|---|\n")
        for s in 单.get("老清单销项") or []:
            f.write("| %s | %s | %s |\n" % (s.get("号"), s.get("销掉"), s.get("说明", "")))
        f.write("\n**结论：%s**\n" % 单.get("结论"))
    打印("\n✅ 全过。成品 → %s" % os.path.relpath(成品, 根))
    打印("   自检报告 → %s" % os.path.relpath(报告, 根))
    return 0


if __name__ == "__main__":
    sys.exit(main())
