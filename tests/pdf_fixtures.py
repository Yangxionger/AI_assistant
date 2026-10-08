"""Small real PDF fixtures built with pypdf; no render/OCR dependencies."""
from io import BytesIO
import textwrap
from pypdf import PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject, TextStringObject
)


def pdf_bytes(page_specs, password=None):
    """Each spec is a dict with optional text/image; None is a blank page."""
    writer = PdfWriter()
    descriptor = DictionaryObject({
        NameObject('/Type'):NameObject('/FontDescriptor'), NameObject('/FontName'):NameObject('/STSong-Light'),
        NameObject('/Flags'):NumberObject(6), NameObject('/FontBBox'):ArrayObject([NumberObject(v) for v in [-250,-200,1000,900]]),
        NameObject('/ItalicAngle'):NumberObject(0), NameObject('/Ascent'):NumberObject(880),
        NameObject('/Descent'):NumberObject(-120), NameObject('/CapHeight'):NumberObject(800), NameObject('/StemV'):NumberObject(80)
    })
    descendant = DictionaryObject({
        NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/CIDFontType0'),
        NameObject('/BaseFont'):NameObject('/STSong-Light'), NameObject('/FontDescriptor'):writer._add_object(descriptor),
        NameObject('/CIDSystemInfo'):DictionaryObject({NameObject('/Registry'):TextStringObject('Adobe'),
                                                      NameObject('/Ordering'):TextStringObject('GB1'), NameObject('/Supplement'):NumberObject(4)}),
        NameObject('/DW'):NumberObject(1000)
    })
    font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type0'),
                             NameObject('/BaseFont'):NameObject('/STSong-Light'), NameObject('/Encoding'):NameObject('/UniGB-UCS2-H'),
                             NameObject('/DescendantFonts'):ArrayObject([writer._add_object(descendant)])})
    font_ref = writer._add_object(font)
    for spec in page_specs:
        page = writer.add_blank_page(width=595, height=842)
        spec = spec or {}
        resources = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font_ref})})
        commands = []
        if spec.get('image'):
            image = DecodedStreamObject()
            image.set_data(bytes([90,140,200])*20*20)
            image.update({NameObject('/Type'):NameObject('/XObject'), NameObject('/Subtype'):NameObject('/Image'),
                          NameObject('/Width'):NumberObject(20), NameObject('/Height'):NumberObject(20),
                          NameObject('/ColorSpace'):NameObject('/DeviceRGB'), NameObject('/BitsPerComponent'):NumberObject(8)})
            resources[NameObject('/XObject')] = DictionaryObject({NameObject('/Im0'):writer._add_object(image)})
            commands.append('q 500 0 0 700 40 70 cm /Im0 Do Q')
        if spec.get('text'):
            commands.append('BT /F1 12 Tf 40 790 Td 16 TL')
            for line in spec['text'].splitlines():
                for wrapped in textwrap.wrap(line, width=36, break_long_words=True, break_on_hyphens=False) or ['']:
                    commands.append('<'+wrapped.encode('utf-16-be').hex()+'> Tj T*')
            commands.append('ET')
        if commands:
            stream = DecodedStreamObject()
            stream.set_data(('\n'.join(commands)+'\n').encode('ascii'))
            page[NameObject('/Contents')] = writer._add_object(stream)
        page[NameObject('/Resources')] = resources
    if password:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


TEXT_SPECS = [
    {'text':'星砂学习同步协议的刷新口令是星砂-5831。\n缓存有效期为43分钟。\n这是项目PDF上传测试的独有计算机学习资料。'},
    None,
    {'text':'星砂协议的课程索引重建步骤。\n'+'缓存过期后先清理旧索引，再读取新的课程资料。\n'*12}
]
