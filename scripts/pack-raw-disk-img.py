#!/usr/bin/env python3
"""Pack the raw eMMC image with the vendor-compatible boot layout.

Layout (mirrors the firmware layout this box already boots):
  0-12MB      bootloader.bin  (idblock + ddr + uboot + trust, written as-is)
  12-16MB     misc.bin        (vendor BCB block)
  16-144MB    zeros           (#3 boot / #4 kernel partitions left empty)
  144-176MB   env.bin         (32MB U-Boot env carrying the extlinux boot chain)
  176-177MB   #6 rootfs       (placeholder, zeros)
  177-227MB   #7 boot         (boot.img: FAT32 with extlinux + kernel + slack)
  GPT: stock vendor entry table (embedded below) with #6/#7 re-ranged,
       primary at LBA1, backup at the tail. No protective MBR (stock also
       ships without one). Total 464929 LBAs (~227MB).

The boot chain this enables (matches the shipped env):
  bootcmd -> test -e mmc 0:7 /boot/extlinux/extlinux.conf
          -> sysboot mmc 0:7 ... /boot/extlinux/extlinux.conf

Usage:
  pack-raw-disk-img.py <bootloader.bin> <boot.img> <output.img>
                       [--env env.bin.xz] [--misc misc.bin.xz]

The env/misc inputs accept raw or .xz payloads and default to the vendor
files bundled under tools/rkbin/rk3566/wxy-oect/.
"""
import argparse
import lzma
import os
import struct
import sys
import zlib

SECTOR = 512
BOOTLOADER_MAX = 12 << 20          # 0-12MB bootloader block
MISC_OFF, MISC_SIZE = 12 << 20, 4 << 20
ENV_OFF, ENV_SIZE = 144 << 20, 32 << 20
ROOTFS_LBA, ROOTFS_LEN_LBA = 360448, 2048          # #6: 176MB, 1MB placeholder
P7_LBA, P7_LEN_LBA = 362496, 102400                # #7: 177MB, 50MB
DISK_LAST_LBA = P7_LBA + P7_LEN_LBA - 1 + 33       # +32 backup entries +1 hdr
TOTAL = (DISK_LAST_LBA + 1) * SECTOR               # ~227MB
ENT_BAK_LBA = DISK_LAST_LBA - 32

# Stock vendor GPT, extracted verbatim from the firmware's head block
# (LBA1 header + LBA2 entries). Entries #1-#5 are used as-is; #6/#7 are
# re-ranged; the disk GUID is kept.
STOCK_HDR92 = bytes.fromhex(
    "4546492050415254000001005c0000002373ddf5000000000100000000000000"
    "ffffe800000000002200000000000000deffe8000000000000001a4900002645"
    "800019ab00004c4a02000000000000008000000080000000a74e49d1")
STOCK_ENTRIES = [bytes.fromhex(h) for h in (
    # 1 uboot  8-12MB
    "00006f200000314b8000495e000029a600007d6d00003a4f80005e2b0000328d0040000000000000ff5f0000000000000000000000000000750062006f006f0074000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    # 2 misc    12-16MB
    "00001ac90000554a8000324c000019060000780d000069458000243900000b5d0060000000000000ff7f00000000000000000000000000006d0069007300630000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    # 3 boot    16-80MB (left empty; sysboot reads #7 instead)
    "0000765100007e49800069ab00000018000073ef00000f4f80005d9b00000a030080000000000000ff7f020000000000000000000000000062006f006f00740000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    # 4 kernel  80-144MB (left empty)
    "00001d8000005840800051430000242400003ddc0000284480005f1a000005c70080020000000000ff7f04000000000000000000000000006b00650072006e0065006c00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    # 5 env     144-176MB
    "00000e1e00003146800034db00006d720000079c0000564e8000657e000078f00080040000000000ff7f050000000000000000000000000065006e007600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    # 6 rootfs  (re-ranged to the 1MB placeholder)
    "00002ef200007e4480003d3d00006a4a00004e610000534b80001d28000054a90080050000000000ff7f250000000000000000000000000072006f006f00740066007300000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    # 7 (boot_a; re-ranged to hold boot.img)
    "af3dc60f838472478e793d69d8477de4349d9e785ee846859a656e660dc5f0cb0080250000000000ff0f2700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
)]


def die(msg):
    sys.exit("error: %s" % msg)


def load_blob(path, want_size, what):
    if not os.path.exists(path):
        die("%s: %s is missing" % (what, path))
    op = lzma.open(path, "rb") if path.endswith(".xz") else open(path, "rb")
    with op as f:
        data = f.read()
    if want_size is not None and len(data) != want_size:
        die("%s: %s is %d bytes, expected %d" % (what, path, len(data), want_size))
    return data


def gpt_header(cur, bk, first, last, ent_lba, disk_guid, ent_crc):
    h = bytearray(b"EFI PART" + struct.pack("<IIIIQQQQ", 0x10000, 92, 0, 0,
                                            cur, bk, first, last))
    h += disk_guid + struct.pack("<QII", ent_lba, 128, 128) + \
        struct.pack("<I", ent_crc)
    assert len(h) == 92
    struct.pack_into("<I", h, 16, zlib.crc32(bytes(h)) & 0xFFFFFFFF)
    return bytes(h)


def build_gpt():
    ents = [bytearray(e) for e in STOCK_ENTRIES]
    struct.pack_into("<QQ", ents[5], 32, ROOTFS_LBA, ROOTFS_LBA + ROOTFS_LEN_LBA - 1)
    struct.pack_into("<QQ", ents[6], 32, P7_LBA, P7_LBA + P7_LEN_LBA - 1)
    table = bytearray().join(bytes(e) for e in ents)
    table += bytes((128 - len(ents)) * 128)
    assert len(table) == 128 * 128
    ent_crc = zlib.crc32(bytes(table)) & 0xFFFFFFFF

    stock = STOCK_HDR92
    disk_guid = stock[56:72]
    first_usable = 34
    last_usable = DISK_LAST_LBA - 33
    primary = gpt_header(1, DISK_LAST_LBA, first_usable, last_usable, 2,
                         disk_guid, ent_crc)
    backup = gpt_header(DISK_LAST_LBA, 1, first_usable, last_usable,
                        ENT_BAK_LBA, disk_guid, ent_crc)
    return primary, backup, bytes(table)


def self_verify(path, env_blob, boot_blob):
    """Re-read the finished image and validate every load-bearing byte."""
    with open(path, "rb") as f:
        img = f.read()
    if len(img) != TOTAL:
        die("self-check: image is %d bytes, expected %d" % (len(img), TOTAL))
    hdr = img[SECTOR:2 * SECTOR]
    if hdr[:8] != b"EFI PART":
        die("self-check: no GPT at LBA1")
    stored = struct.unpack_from("<I", hdr, 16)[0]
    probe = bytearray(hdr[:92])
    struct.pack_into("<I", probe, 16, 0)
    if zlib.crc32(bytes(probe)) & 0xFFFFFFFF != stored:
        die("self-check: primary GPT header CRC mismatch")
    ent_lba, nent, esz = struct.unpack_from("<QII", hdr, 72)
    ent = img[ent_lba * SECTOR:ent_lba * SECTOR + nent * esz]
    if zlib.crc32(ent) & 0xFFFFFFFF != struct.unpack_from("<I", hdr, 88)[0]:
        die("self-check: GPT entry CRC mismatch")
    for idx, want in ((5, (ROOTFS_LBA, ROOTFS_LBA + ROOTFS_LEN_LBA - 1)),
                      (6, (P7_LBA, P7_LBA + P7_LEN_LBA - 1))):
        got = struct.unpack_from("<QQ", ent, idx * esz + 32)
        if got != want:
            die("self-check: partition #%d range %s != %s" % (idx + 1, got, want))
    if img[ENV_OFF:ENV_OFF + len(env_blob)] != env_blob:
        die("self-check: env region mismatch at %d" % ENV_OFF)
    p7_off = P7_LBA * SECTOR
    if img[p7_off:p7_off + len(boot_blob)] != boot_blob:
        die("self-check: boot.img mismatch in partition #7")
    if img[64 * SECTOR:64 * SECTOR + 4] != b"RKSS":
        die("self-check: IDBlock magic missing at sector 64")
    print("self-check OK: GPT CRCs, #6/#7 ranges, env @144MB, boot.img @%dMB"
          % (P7_LBA * SECTOR >> 20))


def main():
    ap = argparse.ArgumentParser(description="vendor-compatible raw eMMC packer")
    ap.add_argument("bootloader")
    ap.add_argument("bootimg")
    ap.add_argument("output")
    here = os.path.dirname(os.path.abspath(__file__))
    vend = os.path.join(here, "..", "tools", "rkbin", "rk3566", "wxy-oect")
    ap.add_argument("--env", default=os.path.join(vend, "env.bin.xz"))
    ap.add_argument("--misc", default=os.path.join(vend, "misc.bin.xz"))
    a = ap.parse_args()

    bl = load_blob(a.bootloader, None, "bootloader")
    if len(bl) > BOOTLOADER_MAX:
        die("bootloader: %s is %d bytes, max %d" % (a.bootloader, len(bl), BOOTLOADER_MAX))
    boot = load_blob(a.bootimg, None, "boot.img")
    if len(boot) > P7_LEN_LBA * SECTOR:
        die("boot.img: %s is %d bytes, partition #%d holds %d"
            % (a.bootimg, len(boot), 7, P7_LEN_LBA * SECTOR))
    env = load_blob(a.env, ENV_SIZE, "env")
    misc = load_blob(a.misc, MISC_SIZE, "misc")

    primary, backup, table = build_gpt()
    with open(a.output, "wb") as out:
        out.truncate(TOTAL)
        out.seek(0)
        out.write(bl)
        out.seek(MISC_OFF)
        out.write(misc)
        out.seek(ENV_OFF)
        out.write(env)
        out.seek(P7_LBA * SECTOR)
        out.write(boot)
        out.seek(SECTOR)
        out.write(primary)
        out.seek(2 * SECTOR)
        out.write(table)
        out.seek(ENT_BAK_LBA * SECTOR)
        out.write(table)
        out.seek(DISK_LAST_LBA * SECTOR)
        out.write(backup)

    self_verify(a.output, env, boot)
    print("packed %s: %d bytes (%.0f MiB)" % (a.output, TOTAL, TOTAL >> 20))
    print("  bootloader @0 (%d bytes), misc @12MB, env @144MB,"
          % len(bl))
    print("  partition #7 boot @%dMB (%d bytes of %dMB), GPT backup @LBA %d"
          % (P7_LBA * SECTOR >> 20, len(boot), P7_LEN_LBA * SECTOR >> 20, DISK_LAST_LBA))


if __name__ == "__main__":
    main()
