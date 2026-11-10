"""Second pass over the stage-1 corpus, organised by ``V4/mesale_khata.txt``.

Every rule carries the code of one taxonomy category: ``E*`` spelling and
vocabulary, ``P*`` punctuation and spacing, ``S*`` scientific style.  Categories
a deterministic rule can repair are rewritten; categories that need judgement
(paraphrase, pronoun ambiguity, redundancy across sentences) are only counted,
so the numbers state exactly what is left for the language-model pass.

No rule merges, splits or drops a sentence: the corpus shape is untouched.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

ZWNJ = "\u200c"

PERSIAN = "\u0600-\u06ff"
#: Persian letters only.  The block also holds «، ؛ ؟ ۴» and the combining
#: marks, so «Persian character» is not the same as «letter».
LETTERS = "\u0621-\u063f\u0641-\u064a\u0671\u067e\u0686\u0698\u06a9\u06af\u06be\u06c1\u06cc\u06d5"
DIGITS = "0-9\u0660-\u0669\u06f0-\u06f9"
HAMZA_ABOVE = "\u0654"
SILENT_HE = "\u0647"

#: Vowel marks and decoration.  U+0654 (hamza above, the Persian ezafe) is
#: excluded on purpose: that is the mark the rules *add*.
HARAKAT_CLASS = "\u064b-\u0652\u0653\u0655\u0670"

_EDGE_CHARS = "()[]{}«»“”\"'`.,;:!?؟،؛·…-_–—/|*+×=<>&%"

#: A collective noun only governs a singular verb when nothing is attached to
#: it, so a half-space counts as part of the word here.
_COLLECTIVE_EDGE = f"{PERSIAN}{ZWNJ}"


def Z(text: str) -> str:
    """Expand ``~`` into a half-space so the tables below stay readable."""
    return text.replace("~", ZWNJ)


CATEGORIES: dict[str, str] = {
    "E1": "کاربرد واژهٔ هم‌آوا یا متشابه با معنای نادرست",
    "E2": "غلط املایی متداول",
    "E3": "مغایرت با رسم‌الخط معیار",
    "E4": "جمع بستن واژهٔ فارسی با نشانهٔ عربی",
    "E5": "خطای تای گرد (ة) در واژه‌های عربی",
    "E5r": "ة باقی‌مانده (نقل عربی، عمداً دست‌نخورده)",
    "E6": "خطای الف مقصوره در واژه‌های عربی",
    "E7": "خطای همزهٔ میانی",
    "E8": "خطای همزهٔ پایانی",
    "E9": "خطای کرسی همزه",
    "E9r": "کرسی همزهٔ باقی‌مانده (شکل پذیرفتهٔ «سوال»)",
    "E10": "تشدیدگذاری نادرست یا غیرضروری",
    "E10r": "تشدید باقی‌مانده (نیازمند مدل)",
    "E11": "حرکت‌گذاری نادرست یا غیرضروری / یای میانجی",
    "E11r": "حرکت باقی‌مانده (راهنمای تلفظ، عمداً نگه‌داشته)",
    "E12": "ساخت قید نامأنوس با «به + صفت»",
    "P1": "حذف نیم‌فاصله در پیشوند فعلی",
    "P2": "حذف نیم‌فاصله در نشانهٔ جمع",
    "P3": "حذف نیم‌فاصله در پسوندهای تفضیلی و عالی",
    "P4": "حذف نیم‌فاصله در واژه‌های مرکب",
    "P5": "فاصلهٔ کامل به‌جای نیم‌فاصله (شناسهٔ پیوندی)",
    "P6": "پیوسته‌نویسی کلمات مستقل",
    "P7": "جدانویسی نادرست اجزای وابسته",
    "P8": "فاصلهٔ نادرست پیش از نشانهٔ نگارشی",
    "P9": "فاصلهٔ نادرست پس از نشانهٔ نگارشی",
    "P10": "فاصله‌گذاری نادرست پرانتز",
    "P11": "فاصله‌گذاری نادرست کروشه",
    "P12": "کاربرد فعل محاوره‌ای در متن رسمی",
    "P13": "کاربرد صورت محاوره‌ای واژه در متن رسمی",
    "P14": "حذف یا نگذاشتن ویرگول در جایگاه‌های ویرگول‌طلب",
    "P15": "نقطه‌گذاری نادرست پایان جمله",
    "P16": "حذف یا نادرست‌گذاشتن دونقطه",
    "S1": "کاربرد ضمیر اول‌شخص مفرد در متن علمی",
    "S2": "کاربرد ضمیر اول‌شخص جمع در متن علمی",
    "S3": "استفادهٔ بی‌مورد از واژهٔ لاتین در متن اصلی",
    "S4": "استفاده از واژهٔ خارجی با وجود معادل فارسی",
    "S4r": "واژهٔ خارجی باقی‌مانده (نیازمند مدل)",
    "S5": "نوشتن نام اشخاص خارجی با خط لاتین در متن فارسی",
    "S6": "کاربرد لقب پیش از نام پژوهشگر",
    "S7": "ابهام مرجع ضمیر اشاره",
    "S8": "ابهام مرجع ضمیر شخصی",
    "S9": "تکرار بیش‌ازحد ضمایر اشاره",
    "S10": "تکرار بیش‌ازحد ضمیر «وی»",
    "S11": "حشو معنایی",
    "S12": "تکرار مکررات",
    "S13": "تکرار واژگانی بیش‌ازحد",
    "S14": "عدم تطابق نهاد و فعل",
    "S14r": "موارد باقی‌ماندهٔ S14 (نیازمند مدل)",
    "S15": "عدم تطابق نهاد و صفت یا مسند",
    "S15r": "موارد باقی‌ماندهٔ S15 (نیازمند مدل)",
    "S16": "کاربرد نامناسب ساخت مجهول یا غیرشخصی",
}

#: Categories this pass only measures; repairing them needs a model.
MEASURED_ONLY = frozenset(
    {
        "E1",
        "E5r",
        "E9r",
        "E10r",
        "E11r",
        "E12",
        "P14",
        "S1",
        "S2",
        "S3",
        "S4r",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
        "S12",
        "S13",
        "S14r",
        "S15r",
        "S16",
    }
)

# ---------------------------------------------------------------------------
# Word tables: exact corpus token -> standard form.  Only pairs whose wrong
# form is a mistake and never a legitimate variant may appear here.
# ---------------------------------------------------------------------------

_WORD_TABLES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "E2",
        {
            "توجیح": "توجیه",
            "باطری": "باتری",
            "مزایقه": "مضایقه",
            "وحله": "وهله",
            "ملتفه": "مختلفه",
            "موزد": "مورد",
            "مقاوله": "مقابله",
            "صحهت": "صحت",
            "المان": "آلمان",
            "المانی": "آلمانی",
            "موزائیک": "موزاییک",
            "دایم": "دائم",
            "قائده": "قاعده",
            "حوضه": "حوزه",
        },
    ),
    (
        "E3",
        {
            "هیچکس": Z("هیچ~کس"),
            "علاقمند": Z("علاقه~مند"),
            "بدلیل": Z("به~دلیل"),
            "بتدریج": Z("به~تدریج"),
            "بطور": Z("به~طور"),
            "بطوری": Z("به~طوری"),
        },
    ),
    (
        "E4",
        {
            "گزارشات": Z("گزارش~ها"),
            "پژوهشات": Z("پژوهش~ها"),
            "گرایشات": Z("گرایش~ها"),
            "بازرسین": "بازرسان",
            "مدیرین": "مدیران",
            "مسئولین": "مسئولان",
            "مسولین": "مسئولان",
            "کشاورزین": "کشاورزان",
            "دبیرین": "دبیران",
            "مجرمین": "مجرمان",
            "پیامات": Z("پیام~ها"),
            "خانهات": Z("خانه~ها"),
            "ماشینات": Z("ماشین~ها"),
            "بیمارستانات": Z("بیمارستان~ها"),
            "مدرسهات": Z("مدرسه~ها"),
            "اندیشهات": Z("اندیشه~ها"),
            "امتیازات": Z("امتیاز~ها"),
            "بازدیدات": Z("بازدید~ها"),
            "فرهنگات": Z("فرهنگ~ها"),
            "پاسخات": Z("پاسخ~ها"),
        },
    ),
    (
        "E6",
        {
            "حتا": "حتی",
            "عیسا": "عیسی",
            "کبرا": "کبری",
            "صغرا": "صغری",
        },
    ),
    (
        "E7",
        {
            "مسله": "مسئله",
            "مسئلهات": "مسائل",
            "مسول": "مسئول",
            "مسولیت": "مسئولیت",
            "مسولانه": "مسئولانه",
            "ریس": "رئیس",
            "متاسف": "متأسف",
            "متاسفانه": "متأسفانه",
            "متاسفم": "متأسفم",
            "موثر": "مؤثر",
            "موثره": "مؤثره",
            "موسسه": "مؤسسه",
            "موسسات": "مؤسسات",
            "مولف": "مؤلف",
            "مومن": "مؤمن",
            "مومنان": "مؤمنان",
            "موید": "مؤید",
        },
    ),
    (
        "E8",
        {
            "امضاء": "امضا",
            "اجراء": "اجرا",
            "استثناء": "استثنا",
            "جزاء": "جزا",
            "اشتراء": "اشترا",
            "شی": "شیء",
        },
    ),
    (
        "E9",
        {
            "سئوال": "سؤال",
            "سئوالها": Z("سؤال~ها"),
            "سئوالات": "سؤالات",
            "مطمن": "مطمئن",
            "هیت": "هیئت",
            "سائر": "سایر",
        },
    ),
    (
        "E10",
        {
            "مَبنّی": "مبنی",
            "مبنّی": "مبنی",
            "مُلغّی": "ملغی",
            "ملغّی": "ملغی",
        },
    ),
    (
        "P6",
        {
            "بدست": "به دست",
            "همانوقت": "همان وقت",
            "آنوقت": "آن وقت",
            "اینوقت": "این وقت",
            "اینطور": Z("این~طور"),
            "همینطور": Z("همین~طور"),
            "آنطور": Z("آن~طور"),
            "همانطور": Z("همان~طور"),
            "آنها": Z("آن~ها"),
            "اینها": Z("این~ها"),
            "دراین": "در این",
            "درآن": "در آن",
            "درهر": "در هر",
            "آنرا": "آن را",
            "اینرا": "این را",
        },
    ),
    (
        "P12",
        {
            "میشه": Z("می~شود"),
            "نمیشه": Z("نمی~شود"),
            "میشد": Z("می~شد"),
            "میشود": Z("می~شود"),
            "میشوند": Z("می~شوند"),
            "نمیشود": Z("نمی~شود"),
            "میکرد": Z("می~کرد"),
            "میکند": Z("می~کند"),
            "میکنه": Z("می~کند"),
            "میکنند": Z("می~کنند"),
            "نمیکند": Z("نمی~کند"),
            "میگه": Z("می~گوید"),
            "نمیگه": Z("نمی~گوید"),
            "میگم": Z("می~گویم"),
            "میگفت": Z("می~گفت"),
            "میگردد": Z("می~گردد"),
            "میرود": Z("می~رود"),
            "نمیرود": Z("نمی~رود"),
            "میرسد": Z("می~رسد"),
            "میداد": Z("می~داد"),
            "میدهد": Z("می~دهد"),
            "میده": Z("می~دهد"),
            "نمیده": Z("نمی~دهد"),
            "میداند": Z("می~داند"),
            "نمیداند": Z("نمی~داند"),
            "میتواند": Z("می~تواند"),
            "نمیتواند": Z("نمی~تواند"),
            "میتونه": Z("می~تواند"),
            "نمیتونه": Z("نمی~تواند"),
            Z("می~تونه"): Z("می~تواند"),
            Z("نمی~تونه"): Z("نمی~تواند"),
            Z("می~خواد"): Z("می~خواهد"),
            Z("نمی~خواد"): Z("نمی~خواهد"),
            Z("می~دونم"): Z("می~دانم"),
            Z("می~دونی"): Z("می~دانی"),
            Z("می~زنه"): Z("می~زند"),
            Z("می~بره"): Z("می~برد"),
            Z("می~ره"): Z("می~رود"),
            Z("نمی~ره"): Z("نمی~رود"),
            Z("می~نویسه"): Z("می~نویسد"),
            "داره": "دارد",
            "دارن": "دارند",
            "میرن": Z("می~روند"),
            "میگن": Z("می~گویند"),
            "میخوان": Z("می~خوانند"),
            "میخواد": Z("می~خواهد"),
            "نمخواد": Z("نمی~خواهد"),
            "نمیخواد": Z("نمی~خواهد"),
            "میباشد": Z("می~باشد"),
            "میاید": Z("می~آید"),
            "میآمد": Z("می~آمد"),
            "میبیند": Z("می~بیند"),
            "نمیبیند": Z("نمی~بیند"),
            "خواهدشد": "خواهد شد",
        },
    ),
    (
        "P13",
        {
            "اونا": Z("آن~ها"),
            "اینا": Z("این~ها"),
            "خونه": "خانه",
            "آروم": "آرام",
            "اونجا": "آنجا",
            "دیگه": "دیگر",
            Z("یواش~یواش"): Z("آرام~آرام"),
            "یواش": "آهسته",
        },
    ),
    (
        "S4",
        {
            "متدولوژی": Z("روش~شناسی"),
            "استراتژی": "راهبرد",
            "تکنولوژی": "فناوری",
            "کمپانی": "شرکت",
            "میتینگ": "جلسه",
            "فیدبک": "بازخورد",
            "ریسرچ": "پژوهش",
            "آنالیز": "تحلیل",
            "آپدیت": Z("به~روزرسانی"),
            "اپدیت": Z("به~روزرسانی"),
            "دانلود": "بارگیری",
            "پلاگین": "افزونه",
            "فریمورک": "چارچوب",
            "کانسپت": "مفهوم",
            "لوکیشن": "موقعیت",
            "کلاب": "باشگاه",
            "پروسه": "فرآیند",
            "موزیک": "موسیقی",
            "فایل": "پرونده",
            "فرمت": "قالب",
            "پسورد": "گذرواژه",
            "دیتا": "داده",
        },
    ),
)

# ---------------------------------------------------------------------------
# Phrase tables: multi-word forms repaired in one pass
# ---------------------------------------------------------------------------

_PHRASE_TABLES: tuple[tuple[str, dict[str, str]], ...] = (
    (
        "E8",
        {
            "سو استفاده": "سوءاستفاده",
            "سو ظن": "سوءظن",
            "سو تفاهم": "سوءتفاهم",
            "سو رفتار": "سوءرفتار",
            "سو قصد": "سوءقصد",
            "سو نیت": "سوءنیت",
        },
    ),
    (
        "P4",
        {
            "دانش آموز": Z("دانش~آموز"),
            "دایره المعارف": Z("دایره~المعارف"),
            "دایرة المعارف": Z("دایره~المعارف"),
            "عکس العمل": Z("عکس~العمل"),
            "چشم انداز": Z("چشم~انداز"),
            "سرمایه دار": Z("سرمایه~دار"),
            "هفته نامه": Z("هفته~نامه"),
            "ماه نامه": Z("ماه~نامه"),
            "سه شنبه": Z("سه~شنبه"),
            "بهره وری": Z("بهره~وری"),
            "بهره برداری": Z("بهره~برداری"),
            "زبان شناسی": Z("زبان~شناسی"),
            "سبک شناسی": Z("سبک~شناسی"),
            "نام شناسی": Z("نام~شناسی"),
            "خاک شناسی": Z("خاک~شناسی"),
            "زیست شناسی": Z("زیست~شناسی"),
            "گروه بندی": Z("گروه~بندی"),
            "طبقه بندی": Z("طبقه~بندی"),
            "بخش بندی": Z("بخش~بندی"),
            "نام گذاری": Z("نام~گذاری"),
            "شماره گذاری": Z("شماره~گذاری"),
            "برنامه ریزی": Z("برنامه~ریزی"),
            "پیش بینی": Z("پیش~بینی"),
            "پس انداز": Z("پس~انداز"),
            "علاقه مندی": Z("علاقه~مندی"),
            "مردم سالاری": Z("مردم~سالاری"),
            "همه جا": Z("همه~جا"),
            "هیچ جا": Z("هیچ~جا"),
            "خوش آمد": Z("خوش~آمد"),
        },
    ),
    (
        "P7",
        {
            "خوش حال": "خوشحال",
            "پنج شنبه": "پنجشنبه",
            "چهار شنبه": "چهارشنبه",
            "یک شنبه": "یکشنبه",
            "دو شنبه": "دوشنبه",
            "روز نامه": "روزنامه",
            "کتاب خانه": "کتابخانه",
            "دانش گاه": "دانشگاه",
            "آشپز خانه": "آشپزخانه",
            "در آمد": "درآمد",
            "کار آفرینی": "کارآفرینی",
            "هر چه": "هرچه",
            "این جا": "اینجا",
            "آن جا": "آنجا",
            "هر جا": "هرجا",
        },
    ),
    (
        "S4",
        {
            "اپلیکیشن": "برنامه کاربردی",
            "اپلیکیشن‌ها": Z("برنامه~کاربردی~ها"),
        },
    ),
    (
        "S11",
        {
            "به همراه با": "به همراه",
            Z("به~همراه با"): Z("به~همراه"),
            "در بین": "بین",
            "بسیار خیلی": "بسیار",
            "خیلی بسیار": "بسیار",
            "تقریباً حدود": "تقریباً",
            "حدود تقریباً": "تقریباً",
            "گذشتهٔ پیشین": "گذشته",
            "همچنین نیز": "همچنین",
            "نیز همچنین": "همچنین",
        },
    ),
)

# ---------------------------------------------------------------------------
# Productive spacing vocabulary
# ---------------------------------------------------------------------------

#: Verbal prefixes written with a full space in the corpus but joined by law.
_VERB_PREFIXES = frozenset({"می", "نمی"})

#: Never pull these behind می/نمی: they are not verb stems.
_PREFIX_STOP = frozenset(
    {
        "من", "ما", "او", "وی", "تو", "شما", "ایشان", "خود", "این", "آن", "اینها", "آن‌ها",
        "همه", "بعضی", "برخی", "حتی", "هم", "بله", "خیر", "چرا", "اگر", "که", "از", "به",
        "با", "در", "بر", "تا", "روی", "همیشه", "هر", "این‌ها", "آنها", "چون", "هرگز",
    }
)

_PLURAL_MARKERS = frozenset({"ها", "های"})
_DEGREE_MARKERS = frozenset({"تر", "ترین"})

#: These two take a full join, not a half-space: بیشتر, کمتر, بیشترین.
_JOINED_COMPARATIVES = frozenset({"بیش", "کم"})

#: Bound person clitics; only joined after a silent he (کرده اند -> کرده‌اند).
_BOUND_CLITICS = frozenset({"ام", "ایم", "اید", "اند"})

# ---------------------------------------------------------------------------
# Compiled rules
# ---------------------------------------------------------------------------

_HARAKAT_RE = re.compile(rf"[{HARAKAT_CLASS}]")
#: Only a word-final هِ is the Persian ezafe; an interior one is a vowelled
#: proper name (هِنری) and must keep its marks.
_EZAFE_RE = re.compile(rf"{SILENT_HE}[\u064d\u0650]$")
_TAA_MID_RE = re.compile(rf"([{PERSIAN}])ة([{PERSIAN}])")
_TAA_END_RE = re.compile(rf"([{PERSIAN}])ة(?![{PERSIAN}])")
_TASHDID_RE = re.compile("[\u0651]")
#: Arabic definite article or tanween: the word is an Arabic form (or a
#: quotation) and its vowel marks are not a Persian ezafe mistake.
_ARABIC_FORM_RE = re.compile(r"^(?:ال)|[\u064b-\u064d]")
_PERSIAN_TAIL_RE = re.compile(rf"[{PERSIAN}]$")
_PERSIAN_HEAD_RE = re.compile(rf"[{PERSIAN}]")

#: Marks that must be followed by a space (P9).
_GAP_PUNCT = "،؛:!?؟.,"
#: A character that may legitimately sit right after a mark without a space.
_NO_GAP_AFTER = frozenset("،؛:!?؟.)]}»”\"'`" + ZWNJ)

#: Only a number split on both sides is unambiguously one decimal; «۱. ۵» may
#: be a sentence end followed by a quantity.
_DECIMAL_GAP_RE = re.compile(rf"([{DIGITS}])[ ]+([.٫])[ ]+([{DIGITS}])")
#: Doubled marks (optionally space-separated) from broken dumps. The dot has
#: its own pattern, matched only on a run of exactly two, so «…» survives.
_DUP_MARK_RE = re.compile(r"([،؛!?])(?: ?\1)+")
_DUP_DOT_RE = re.compile(r"(?<!\.)\.(?: ?\.(?!\.))+")
#: A broken comma in front of the closing dot (or behind it) is one mark: the dot.
_MARK_AROUND_DOT_RE = re.compile(r"(?:[،؛:][ ]*\.(?!\.)|\.[ ]*[،؛])")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ ]+([،؛:!?؟])(?![،؛:!?؟])")
_SPACE_BEFORE_DOT_RE = re.compile(r"[ ]+\.(?!\.)")
_BRACKET_INNER_OPEN_RE = re.compile(r"([(«\[{])[ ]+")
_BRACKET_INNER_CLOSE_RE = re.compile(r"[ ]+([)»\]}])")
_BRACKET_OUTER_OPEN_RE = re.compile(rf"(?<=[{PERSIAN}{DIGITS}A-Za-z])(?=[(«[])")
#: A closing bracket is separated from the *word* that follows it; a following
#: punctuation mark must stay attached.
_BRACKET_OUTER_CLOSE_RE = re.compile(rf"(?<=[)\]»])(?=[{LETTERS}{DIGITS}A-Za-z])")
#: A quoted clause may open with «…» or straight with its own subject; a
#: conjunction (که، تا، …) means reported speech and must not take a colon.
_QUOTED_SUBJECT = "من|ما|تو|شما|او|وی|ایشان|این|آن"
_QUOTE_INTRO_RE = re.compile(
    rf"(?<![{PERSIAN}])(گفت|گفته است|گفتند|می{ZWNJ}گوید|می{ZWNJ}گویند|نوشت|نوشته است|"
    rf"اظهار داشت|بیان کرد|اعلام کرد|یادآوری می{ZWNJ}کند)"
    rf"(?= «| (?=({_QUOTED_SUBJECT})(?![{PERSIAN}])))"
)
_GLUED_COPULA_RE = re.compile(rf"([{PERSIAN}]){ZWNJ}?(?=هستند(?![{PERSIAN}]))")
_REDUNDANT_ATTR_PLURAL_RE = re.compile(
    rf"(?<![{PERSIAN}])([{PERSIAN}]{{2,}}ی{ZWNJ}ها) (?=هستند(?![{PERSIAN}]))"
)
#: «گروه‌های مختلف … شده‌اند» really is plural, so a half-space may not follow
#: the collective noun before the singular-verb rewrite is applied.
_LIST_INTRO_RE = re.compile(rf"(?<![{PERSIAN}])(?:عبارتند از|عبارت است از|به شرح زیر است|به قرار زیر است)(?![{PERSIAN}])")
_COLLECTIVE_RE = re.compile(
    rf"(?<![{_COLLECTIVE_EDGE}])(?:تعداد|گروه|تیم|هیئت|دسته|جمعیت|مجموعه)(?![{_COLLECTIVE_EDGE}])"
)
_PLURAL_PERFECT_RE = re.compile(rf"([{PERSIAN}]{{2,}}){ZWNJ}اند(?![{PERSIAN}])")
_CLAUSE_END_RE = re.compile(r"[.،؛:!?؟]")
#: A sentence that trails off into brackets may already carry its mark inside
#: them, so only a letter or digit justifies adding the final dot.
_TERMINAL_RE = re.compile(rf"[{LETTERS}{DIGITS}A-Za-z]$")
_TITLE_DOT_RE = re.compile(r"[. ]+$")
_MULTI_ZWNJ_RE = re.compile(rf"{ZWNJ}{{2,}}")
_DOUBLE_SPACE_RE = re.compile(r" {2,}")
_LETTER_RE = re.compile(rf"[{LETTERS}A-Za-z]")

_MEASURE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("E1", re.compile(rf"(?<![{PERSIAN}])(?:ثواب|متبوع|انتصاب)(?![{PERSIAN}])")),
    ("E5r", re.compile(rf"ة(?![{PERSIAN}])")),
    ("E9r", re.compile(rf"(?<![{PERSIAN}])سوال(?![{PERSIAN}])")),
    ("E10r", _TASHDID_RE),
    ("E11r", re.compile(rf"[{PERSIAN}][{HARAKAT_CLASS}]")),
    (
        "E12",
        re.compile(
            rf"(?<![{PERSIAN}])(?:به{ZWNJ}طور|به طور|به{ZWNJ}صورت|به صورت|به شکل|بهطوری)(?![{PERSIAN}])"
        ),
    ),
    ("S1", re.compile(rf"(?<![{PERSIAN}])(?:من|خودم)(?![{PERSIAN}])|(?<![{PERSIAN}])(?:هستم|کردم|کنم|گفتم|داشتم|بودم|شدم|خواهم)(?![{PERSIAN}])")),
    ("S2", re.compile(rf"(?<![{PERSIAN}])(?:ما|خودمان)(?![{PERSIAN}])|(?<![{PERSIAN}])(?:هستیم|کردیم|کنیم|گفتیم|داشتیم|بودیم|شویم|خواهیم)(?![{PERSIAN}])")),
    ("S3", re.compile(r"[A-Za-z]{2,}")),
    ("S5", re.compile(r"\b[A-Z][a-z]{2,}\b")),
    (
        "S6",
        re.compile(rf"(?<![{PERSIAN}])(?:دکتر|پروفسور|استاد|مهندس|جناب)(?![{PERSIAN}])"),
    ),
    (
        "S7",
        re.compile(rf"(?:^| )این (?=(?:است|شد|شود|خود|امر|مورد|روند|رویه|نتیجه)(?![{PERSIAN}]))"),
    ),
    ("S8", re.compile(rf"(?:که|تا|و) (?:او|وی)(?![{PERSIAN}])")),
    ("S10", re.compile(rf"(?<![{PERSIAN}])وی(?![{PERSIAN}])")),
    ("S12", re.compile(r"به عبارت دیگر|به بیان دیگر|بدین معنا که|با عبارت دیگر")),
    (
        "S14r",
        re.compile(
            rf"(?<![{_COLLECTIVE_EDGE}])(?:تعداد|گروه|تیم|هیئت|جمعیت|مجموعه|دسته)(?![{_COLLECTIVE_EDGE}])[^.،؛]{{0,60}}?"
            rf"(?:اند|باشند|هستند|کردند|شدند|دادند|گفتند|داشتند)(?![{PERSIAN}])"
        ),
    ),
    ("S16", re.compile(rf"(?<![{PERSIAN}])(?:توسط|به{ZWNJ}وسیلهٔ|به وسیلهٔ|به{ZWNJ}وسیله|به وسیله)(?![{PERSIAN}])")),
    (
        "S4r",
        re.compile(
            rf"(?<![{PERSIAN}])(?:سیستم|سایت|پروژه|کامپیوتر|موبایل|اینترنت|سرویس|"
            rf"تلویزیون|اندروید|ویندوز|پروتکل)(?![{PERSIAN}])"
        ),
    ),
)

_WORD_LOOKUP: dict[str, tuple[str, str]] = {}
for _code, _table in _WORD_TABLES:
    for _wrong, _right in _table.items():
        if _wrong != _right:
            _WORD_LOOKUP[_wrong] = (_code, _right)

_PHRASE_LOOKUP: dict[str, tuple[str, str]] = {}
for _code, _table in _PHRASE_TABLES:
    for _wrong, _right in _table.items():
        if _wrong != _right:
            _PHRASE_LOOKUP[_wrong] = (_code, _right)

_PHRASE_RE = re.compile(
    rf"(?<![{PERSIAN}A-Za-z])("
    + "|".join(re.escape(key) for key in sorted(_PHRASE_LOOKUP, key=len, reverse=True))
    + rf")(?![{PERSIAN}A-Za-z])"
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class TaxonomyStats:
    """Per-category hit counters for one pass."""

    categories: dict[str, int] = field(default_factory=dict)
    forms: dict[str, dict[str, int]] = field(default_factory=dict)
    sentences: int = 0
    changed: int = 0

    def hit(self, code: str, count: int = 1, form: str | None = None) -> None:
        self.categories[code] = self.categories.get(code, 0) + count
        if form:
            bucket = self.forms.setdefault(code, {})
            bucket[form] = bucket.get(form, 0) + 1

    def merge(self, other: "TaxonomyStats") -> None:
        self.sentences += other.sentences
        self.changed += other.changed
        for code, count in other.categories.items():
            self.categories[code] = self.categories.get(code, 0) + count
        for code, forms in other.forms.items():
            bucket = self.forms.setdefault(code, {})
            for form, count in forms.items():
                bucket[form] = bucket.get(form, 0) + count

    def as_dict(self) -> dict[str, object]:
        return {
            "categories": dict(self.categories),
            "forms": {code: dict(forms) for code, forms in self.forms.items()},
            "sentences": self.sentences,
            "changed": self.changed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TaxonomyStats":
        stats = cls(
            categories=dict(payload["categories"] or {}),  # type: ignore[arg-type]
            forms={code: dict(form) for code, form in (payload["forms"] or {}).items()},  # type: ignore[union-attr]
            sentences=int(payload["sentences"]),  # type: ignore[arg-type]
            changed=int(payload["changed"]),  # type: ignore[arg-type]
        )
        return stats

    def render(self, verbose: bool = False) -> str:
        fixed = [(code, n) for code, n in self.categories.items() if code not in MEASURED_ONLY]
        measured = [(code, n) for code, n in self.categories.items() if code in MEASURED_ONLY]
        fixed.sort(key=lambda item: (-item[1], item[0]))
        measured.sort(key=lambda item: (-item[1], item[0]))
        lines = ["* repaired by rule:"]
        lines += [f"  {code:<4} {CATEGORIES[code]:<52} {n:>9}" for code, n in fixed] or ["  (none)"]
        lines.append("* counted, not repaired (needs a model or kept on purpose):")
        lines += [f"  {code:<4} {CATEGORIES[code]:<52} {n:>9}" for code, n in measured] or ["  (none)"]
        lines.append(f"  sentences {self.sentences}, touched {self.changed}")
        if verbose:
            for code, _ in fixed + measured:
                forms = self.forms.get(code)
                if not forms:
                    continue
                top = sorted(forms.items(), key=lambda item: (-item[1], item[0]))[:8]
                lines.append(f"  {code}: " + ", ".join(f"{form}×{n}" for form, n in top))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _unwrap(token: str) -> tuple[str, str, str]:
    """Split a token into leading punctuation, word core, trailing punctuation."""
    start, stop = 0, len(token)
    while start < stop and token[start] in _EDGE_CHARS:
        start += 1
    while stop > start and token[stop - 1] in _EDGE_CHARS:
        stop -= 1
    return token[:start], token[start:stop], token[stop:]


def _fix_letter_forms(word: str, stats: TaxonomyStats) -> str:
    """Apply the character-level spelling rules to one bare word."""
    fixed = word

    if "\u0629" in fixed and not _HARAKAT_RE.search(fixed):
        new = _TAA_MID_RE.sub(rf"\1ه{ZWNJ}\2", fixed)
        new = _TAA_END_RE.sub(r"\1ه", new)
        if new != fixed:
            stats.hit("E5", form=fixed)
            fixed = new

    if _ARABIC_FORM_RE.search(fixed):
        return fixed

    new = _EZAFE_RE.sub(f"{SILENT_HE}{HAMZA_ABOVE}", fixed)
    if new != fixed:
        stats.hit("E11", form=fixed)
        # the ezafe rewrite supersedes the rest of the word's vowel marks
        fixed = _HARAKAT_RE.sub("", new)

    return fixed


def _fix_token(token: str, stats: TaxonomyStats) -> str:
    head, core, tail = _unwrap(token)
    if not core:
        return token
    entry = _WORD_LOOKUP.get(core)
    if entry is not None:
        code, replacement = entry
        stats.hit(code, form=core)
        core = replacement
    else:
        core = _fix_letter_forms(core, stats)
    return head + core + tail


def _join_suffix(previous: str, token: str) -> tuple[str, str] | None:
    """Return the merged token and its category when a suffix must be attached."""
    if not previous or not _PERSIAN_TAIL_RE.search(previous):
        return None
    if token in _PLURAL_MARKERS:
        return Z(f"{previous}~{token}"), "P2"
    if token in _DEGREE_MARKERS:
        if previous in _JOINED_COMPARATIVES:
            return previous + token, "P3"
        return Z(f"{previous}~{token}"), "P3"
    if token in _BOUND_CLITICS and previous.endswith(SILENT_HE):
        return Z(f"{previous}~{token}"), "P5"
    return None


def _join_tokens(tokens: list[str], stats: TaxonomyStats) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if token in _VERB_PREFIXES and following is not None and following not in _PREFIX_STOP:
            if _PERSIAN_HEAD_RE.match(following):
                stats.hit("P1", form=token)
                out.append(Z(f"{token}~{following}"))
                index += 2
                continue
        if out:
            merged = _join_suffix(out[-1], token)
            if merged is not None:
                joined, code = merged
                stats.hit(code, form=token)
                out[-1] = joined
                index += 1
                continue
        out.append(token)
        index += 1
    return out


def _apply_phrases(text: str, stats: TaxonomyStats) -> str:
    def replace(match: re.Match[str]) -> str:
        code, right = _PHRASE_LOOKUP[match.group(1)]
        stats.hit(code, form=match.group(1))
        return right

    return _PHRASE_RE.sub(replace, text)


def _fix_spacing(text: str, stats: TaxonomyStats) -> str:
    gaps = len(_DECIMAL_GAP_RE.findall(text))
    fixed = _DECIMAL_GAP_RE.sub(r"\1\2\3", text)

    # Doubled marks collapse first: «..» would otherwise hide the space in
    # front of it from the pass below, and «، ،» is matched with its space.
    dups = len(_DUP_MARK_RE.findall(fixed)) + len(_DUP_DOT_RE.findall(fixed))
    fixed = _DUP_MARK_RE.sub(r"\1", fixed)
    fixed = _DUP_DOT_RE.sub(".", fixed)

    before = len(_SPACE_BEFORE_PUNCT_RE.findall(fixed)) + len(_SPACE_BEFORE_DOT_RE.findall(fixed))
    fixed = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", fixed)
    fixed = _SPACE_BEFORE_DOT_RE.sub(".", fixed)

    # Closing that gap can leave «،.»; the dot is the mark that counts.
    around = len(_MARK_AROUND_DOT_RE.findall(fixed))
    fixed = _MARK_AROUND_DOT_RE.sub(".", fixed)

    if before or gaps or dups or around:
        stats.hit("P8", before + gaps + dups + around)
    return fixed


def _needs_space_after(text: str, dot_at: int) -> bool:
    """False when the dot closes an abbreviation (ه.ق) or splits a number."""
    left, right = dot_at - 1, dot_at + 1
    if right < len(text) and text[right] in ".٫،,؛:":
        return False
    if left >= 0 and text[left].isdigit() and right < len(text) and text[right].isdigit():
        return False
    left_len = 0
    cursor = left
    while cursor >= 0 and text[cursor] not in " :":
        if text[cursor] == ".":
            return False
        left_len += 1
        cursor -= 1
    right_len = 0
    cursor = right
    while cursor < len(text) and text[cursor] not in " :":
        right_len += 1
        cursor += 1
    return not (left_len <= 2 and right_len <= 2)


def _insert_spaces(text: str, stats: TaxonomyStats) -> str:
    """Restore the one space that must follow every mark (P9)."""
    out: list[str] = []
    inserted = 0
    for index, char in enumerate(text):
        out.append(char)
        if char not in _GAP_PUNCT or index + 1 >= len(text):
            continue
        following = text[index + 1]
        if following == " " or following in _NO_GAP_AFTER:
            continue
        previous = text[index - 1] if index else ""
        if previous.isdigit() and following.isdigit():
            continue
        if char == "." and not _needs_space_after(text, index):
            continue
        out.append(" ")
        inserted += 1
    if inserted:
        stats.hit("P9", inserted)
    return "".join(out)


def _fix_brackets(text: str, stats: TaxonomyStats) -> str:
    fixed = text
    for opener, closer, code in (("(", ")", "P10"), ("[", "]", "P11"), ("«", "»", "P10")):
        pattern_open = re.compile(rf"{re.escape(opener)}[ ]+")
        pattern_close = re.compile(rf"[ ]+{re.escape(closer)}")
        hits = len(pattern_open.findall(fixed)) + len(pattern_close.findall(fixed))
        fixed = pattern_open.sub(opener, fixed)
        fixed = pattern_close.sub(closer, fixed)
        if hits:
            stats.hit(code, hits)

    glued_open = len(_BRACKET_OUTER_OPEN_RE.findall(fixed))
    fixed = _BRACKET_OUTER_OPEN_RE.sub(" ", fixed)
    glued_close = len(_BRACKET_OUTER_CLOSE_RE.findall(fixed))
    fixed = _BRACKET_OUTER_CLOSE_RE.sub(" ", fixed)
    if glued_open + glued_close:
        stats.hit("P10", glued_open + glued_close)
    return fixed


def _split_glued_copula(text: str, stats: TaxonomyStats) -> str:
    """``کوشاهستند`` -> ``کوشا هستند``; the copula is never part of a word."""
    hits = len(_GLUED_COPULA_RE.findall(text))
    if not hits:
        return text
    stats.hit("S15", hits)
    return _GLUED_COPULA_RE.sub(r"\1 ", text)


def _fix_colons(text: str, stats: TaxonomyStats) -> str:
    inserted = 0

    def quote(match: re.Match[str]) -> str:
        nonlocal inserted
        if text[max(0, match.start() - 8) : match.start()].endswith("توان "):
            return match.group(0)  # «می‌توان گفت…» is an idiom, not a quotation
        inserted += 1
        return f"{match.group(0)}:"

    fixed = _QUOTE_INTRO_RE.sub(quote, text)
    for intro in _LIST_INTRO_RE.finditer(fixed):
        tail = fixed[intro.end() :].lstrip(" ")
        if tail.startswith(":") or not tail:
            continue
        if tail.count("،") + tail.count(" و ") >= 1:
            stats.hit("P16", inserted + 1)
            position = intro.end()
            return f"{fixed[:position]}: {fixed[position:].lstrip(' ')}"
    if inserted:
        stats.hit("P16", inserted)
    return fixed


def _fix_agreement(text: str, stats: TaxonomyStats) -> str:
    """تعداد/گروه + a present-perfect plural verb takes the singular in Persian."""
    subject = _COLLECTIVE_RE.search(text)
    if subject is None:
        return text
    clause_end = len(text)
    for separator in _CLAUSE_END_RE.finditer(text, subject.end()):
        clause_end = separator.start()
        break
    window = text[subject.end() : clause_end]
    verb = _PLURAL_PERFECT_RE.search(window)
    if verb is None or verb.start() > 80:
        return text
    start = subject.end() + verb.start()
    stop = subject.end() + verb.end()
    stats.hit("S14", form=verb.group(0))
    return f"{text[:start]}{verb.group(1)} است{text[stop:]}"


def _ensure_terminal(text: str, stats: TaxonomyStats) -> str:
    if not _TERMINAL_RE.search(text):
        return text
    if sum(1 for char in text if _LETTER_RE.fullmatch(char)) < 3:
        return text
    stats.hit("P15")
    return f"{text}."


def _measure(text: str, stats: TaxonomyStats) -> None:
    for code, pattern in _MEASURE_RULES:
        found = len(pattern.findall(text))
        if found:
            stats.hit(code, found)
    tokens = [token for token in text.split(" ") if token]
    if len(tokens) >= 18 and "،" not in text:
        stats.hit("P14")
    if len(tokens) >= 6 and (text.startswith("این ") or text.startswith("آن ")):
        stats.hit("S9")
    words = [token for token in tokens if len(token) >= 3]
    if words and max(Counter(words).values()) >= 4:
        stats.hit("S13")
    residual = len(_REDUNDANT_ATTR_PLURAL_RE.findall(text))
    if residual:
        stats.hit("S15r", residual)


def apply(text: str, stats: TaxonomyStats, *, is_title: bool = False) -> str:
    """Return ``text`` with every taxonomy rule applied; ``stats`` is updated."""
    if not text or not text.strip():
        return text

    tokens = _join_tokens([_fix_token(token, stats) for token in text.split(" ")], stats)
    fixed = " ".join(tokens)
    fixed = _apply_phrases(fixed, stats)
    fixed = _split_glued_copula(fixed, stats)
    fixed = _fix_brackets(fixed, stats)
    fixed = _fix_colons(fixed, stats)
    fixed = _fix_agreement(fixed, stats)
    fixed = _fix_spacing(fixed, stats)
    fixed = _insert_spaces(fixed, stats)

    fixed = _MULTI_ZWNJ_RE.sub(ZWNJ, fixed)
    fixed = _DOUBLE_SPACE_RE.sub(" ", fixed).strip(" " + ZWNJ)
    if not fixed:
        return text

    if is_title:
        stripped = _TITLE_DOT_RE.sub("", fixed)
        if stripped != fixed:
            stats.hit("P15", form="عنوان")
        return stripped

    fixed = _ensure_terminal(fixed, stats)
    _measure(fixed, stats)
    stats.sentences += 1
    if fixed != text:
        stats.changed += 1
    return fixed
