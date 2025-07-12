ALPHABET_TEXT = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"
C1 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
C2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
C3 = "0123456789"
C4 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"

NTOKENS = 2063592
MAX22 = 4194304
MAXGRID4 = 32400

# Hash lookup tables used for hashed callsigns.  Populated via
# :func:`register_callsign`.
_HASH10 = {}
_HASH12 = {}
_HASH22 = {}


def register_callsign(call: str) -> None:
    """Register ``call`` so hashed callsigns can be recovered."""
    call = call.strip().upper()
    _HASH10[ihashcall(call, 10)] = call
    _HASH12[ihashcall(call, 12)] = call
    _HASH22[ihashcall(call, 22)] = call


def ihashcall(call: str, m: int) -> int:
    """Return the ``m`` bit hash used by WSJT-X."""
    table = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"
    call = call.upper().ljust(11)[:11]
    n = 0
    for ch in call:
        n = 38 * n + table.index(ch)
    return ((47055833459 * n) >> (64 - m)) & ((1 << m) - 1)


def _decode_text(bits: str) -> str:
    val = int(bits, 2)
    chars = []
    for _ in range(13):
        val, r = divmod(val, 42)
        chars.append(ALPHABET_TEXT[r])
    return "".join(reversed(chars)).strip()


def _to_grid4(n: int) -> str:
    j1 = n // (18 * 10 * 10)
    n -= j1 * 18 * 100
    j2 = n // (10 * 10)
    n -= j2 * 100
    j3 = n // 10
    j4 = n % 10
    if not (0 <= j1 <= 17 and 0 <= j2 <= 17 and 0 <= j3 <= 9 and 0 <= j4 <= 9):
        raise ValueError("bad grid4")
    return f"{chr(j1+65)}{chr(j2+65)}{j3}{j4}"


def _unpack28(n: int) -> str:
    if n < NTOKENS:
        if n == 0:
            return "DE"
        if n == 1:
            return "QRZ"
        if n == 2:
            return "CQ"
        if n <= 1002:
            return f"CQ_{n-3:03d}"
        if n <= 532443:
            n -= 1003
            i1 = n // (27 * 27 * 27)
            n -= i1 * (27 * 27 * 27)
            i2 = n // (27 * 27)
            n -= i2 * (27 * 27)
            i3 = n // 27
            i4 = n % 27
            s = C4[i1] + C4[i2] + C4[i3] + C4[i4]
            return "CQ_" + s.strip()
    else:
        n -= NTOKENS
        if n < MAX22:
            return f"<{_HASH22.get(n, '...')}>"
        n -= MAX22
        i1 = n // (36 * 10 * 27 * 27 * 27)
        n -= i1 * (36 * 10 * 27 * 27 * 27)
        i2 = n // (10 * 27 * 27 * 27)
        n -= i2 * (10 * 27 * 27 * 27)
        i3 = n // (27 * 27 * 27)
        n -= i3 * (27 * 27 * 27)
        i4 = n // (27 * 27)
        n -= i4 * (27 * 27)
        i5 = n // 27
        i6 = n % 27
        s = (
            C1[i1] + C2[i2] + C3[i3] + C4[i4] + C4[i5] + C4[i6]
        )
        return s.lstrip().rstrip()
    return ""


def decode77(bitstring: str) -> str:
    """Decode a 77-bit FT8 message to text."""
    if len(bitstring) != 77:
        raise ValueError("bitstring must have length 77")

    n3 = int(bitstring[71:74], 2)
    i3 = int(bitstring[74:77], 2)

    if i3 == 0 and n3 == 0:
        return _decode_text(bitstring[:71])

    if i3 == 0 and n3 == 1:
        n28a = int(bitstring[:28], 2)
        n28b = int(bitstring[28:56], 2)
        n10 = int(bitstring[56:66], 2)
        n5 = int(bitstring[66:71], 2)
        call1 = _unpack28(n28a)
        call2 = _unpack28(n28b)
        call3 = f"<{_HASH10.get(n10, '...')}>"
        rpt = 2 * n5 - 30
        crpt = f"{rpt:+03d}"
        return f"{call1} RR73; {call2} {call3} {crpt}"

    if i3 == 0 and n3 in (3, 4):
        n28a = int(bitstring[:28], 2)
        n28b = int(bitstring[28:56], 2)
        ir = int(bitstring[56:57], 2)
        intx = int(bitstring[57:61], 2)
        nclass = int(bitstring[61:64], 2)
        isec = int(bitstring[64:71], 2)
        call1 = _unpack28(n28a)
        call2 = _unpack28(n28b)
        csec = [
            "AB","AK","AL","AR","AZ","BC","CO","CT","DE","EB","EMA","ENY","EPA","EWA","GA","GH",
            "IA","ID","IL","IN","KS","KY","LA","LAX","NS","MB","MDC","ME","MI","MN","MO","MS",
            "MT","NC","ND","NE","NFL","NH","NL","NLI","NM","NNJ","NNY","TER","NTX","NV","OH","OK",
            "ONE","ONN","ONS","OR","ORG","PAC","PR","QC","RI","SB","SC","SCV","SD","SDG","SF","SFL",
            "SJV","SK","SNJ","STX","SV","TN","UT","VA","VI","VT","WCF","WI","WMA","WNY","WPA","WTX",
            "WV","WWA","WY","DX","PE","NB",
        ]
        ntx = intx + 1
        if n3 == 4:
            ntx += 16
        cntx = f"{ntx:02d}{chr(ord('A')+nclass)}"
        sec = csec[isec - 1] if 1 <= isec <= len(csec) else ""
        if ir == 0 and ntx < 10:
            return f"{call1} {call2} {cntx} {sec}".strip()
        if ir == 1 and ntx < 10:
            return f"{call1} {call2} R{cntx} {sec}".strip()
        if ir == 0 and ntx >= 10:
            return f"{call1} {call2} {cntx} {sec}".strip()
        if ir == 1 and ntx >= 10:
            return f"{call1} {call2} R {cntx} {sec}".strip()

    if i3 == 0 and n3 == 5:
        a = int(bitstring[:23], 2)
        b = int(bitstring[23:47], 2)
        c = int(bitstring[47:71], 2)
        return f"{a:06X}{b:06X}{c:06X}".lstrip()

    if i3 in (1, 2):
        n28a = int(bitstring[:28], 2)
        ipa = int(bitstring[28:29], 2)
        n28b = int(bitstring[29:57], 2)
        ipb = int(bitstring[57:58], 2)
        ir = int(bitstring[58:59], 2)
        igrid4 = int(bitstring[59:74], 2)
        call1 = _unpack28(n28a)
        call2 = _unpack28(n28b)
        if ipa:
            suff = "/R" if i3 == 1 else "/P"
            idx = call1.find(" ")
            if idx >= 3:
                call1 = call1[:idx] + suff
        if ipb:
            suff = "/R" if i3 == 1 else "/P"
            idx = call2.find(" ")
            if idx >= 3:
                call2 = call2[:idx] + suff
        if igrid4 <= MAXGRID4:
            grid = _to_grid4(igrid4)
            if ir:
                return f"{call1} {call2} R {grid}"
            else:
                return f"{call1} {call2} {grid}"
        else:
            irpt = igrid4 - MAXGRID4
            if irpt == 1:
                return f"{call1} {call2}"
            if irpt == 2:
                return f"{call1} {call2} RRR"
            if irpt == 3:
                return f"{call1} {call2} RR73"
            if irpt == 4:
                return f"{call1} {call2} 73"
            isnr = irpt - 35
            if isnr > 50:
                isnr -= 101
            crpt = f"{isnr:+03d}".replace("+0", "+").replace("-0", "-")
            if ir:
                return f"{call1} {call2} R{crpt}"
            else:
                return f"{call1} {call2} {crpt}"

    if i3 == 3:
        itu = int(bitstring[:1], 2)
        n28a = int(bitstring[1:29], 2)
        n28b = int(bitstring[29:57], 2)
        ir = int(bitstring[57:58], 2)
        irpt = int(bitstring[58:61], 2)
        nexch = int(bitstring[61:74], 2)
        call1 = _unpack28(n28a)
        call2 = _unpack28(n28b)
        crpt = f"5{irpt+2}9"
        if nexch > 8000:
            mults = [
                "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS",
                "KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND",
                "OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","NB","NS","QC",
                "ON","MB","SK","AB","BC","NWT","NF","LB","NU","YT","PEI","DC","DR","FR","GD","GR","OV",
                "ZH","ZL","X01","X02","X03","X04","X05","X06","X07","X08","X09","X10","X11","X12","X13",
                "X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X24","X25","X26","X27","X28",
                "X29","X30","X31","X32","X33","X34","X35","X36","X37","X38","X39","X40","X41","X42","X43",
                "X44","X45","X46","X47","X48","X49","X50","X51","X52","X53","X54","X55","X56","X57","X58",
                "X59","X60","X61","X62","X63","X64","X65","X66","X67","X68","X69","X70","X71","X72","X73",
                "X74","X75","X76","X77","X78","X79","X80","X81","X82","X83","X84","X85","X86","X87","X88",
                "X89","X90","X91","X92","X93","X94","X95","X96","X97","X98","X99",
            ]
            mult = mults[nexch - 8001] if nexch - 8001 < len(mults) else ""
            if ir:
                return f"{'TU; ' if itu else ''}{call1} {call2} R {crpt} {mult}".strip()
            else:
                return f"{'TU; ' if itu else ''}{call1} {call2} {crpt} {mult}".strip()
        else:
            serial = nexch
            if ir:
                return f"{'TU; ' if itu else ''}{call1} {call2} R {crpt} {serial:04d}".strip()
            else:
                return f"{'TU; ' if itu else ''}{call1} {call2} {crpt} {serial:04d}".strip()

    if i3 == 4:
        n12 = int(bitstring[:12], 2)
        n58 = int(bitstring[12:70], 2)
        iflip = int(bitstring[70:71], 2)
        nrpt = int(bitstring[71:73], 2)
        icq = int(bitstring[73:74], 2)
        call_chars = []
        tmp = n58
        for _ in range(11):
            call_chars.append(" 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/"[tmp % 38])
            tmp //= 38
        c11 = "".join(reversed(call_chars)).lstrip()
        call_h = _HASH12.get(n12, '...')
        if iflip == 0:
            call1 = f"<{call_h}>" if call_h != '...' else '<...>'
            call2 = c11.strip() or ""
        else:
            call1 = c11.strip()
            call2 = f"<{call_h}>" if call_h != '...' else '<...>'
        if icq:
            return f"CQ {call2}".strip()
        suffix = {0:"",1:" RRR",2:" RR73",3:" 73"}.get(nrpt,"")
        return f"{call1} {call2}{suffix}".strip()

    raise ValueError("Unsupported message type")

