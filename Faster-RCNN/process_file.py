import xml.dom.minidom as xmldom


def read_xml(file):
    xml_file = xmldom.parse(file)

    # level 1 tag
    ele = xml_file.documentElement

    img_info = {}
    get_wid = ele.getElementsByTagName('width')
    wid = get_wid[0].firstChild.data
    img_info['width'] = wid
    get_hei = ele.getElementsByTagName('height')
    hei = get_hei[0].firstChild.data
    img_info['height'] = hei

    get_obj = ele.getElementsByTagName('object')
    objs = []
    for o in get_obj:
        obj = {}
        get_name = o.getElementsByTagName('name')
        name = get_name[0].firstChild.data
        obj['name'] = name

        get_xmin = o.getElementsByTagName('xmin')
        xmin = get_xmin[0].firstChild.data
        obj['xmin'] = xmin

        get_ymin = o.getElementsByTagName('ymin')
        ymin = get_ymin[0].firstChild.data
        obj['ymin'] = ymin

        get_xmax = o.getElementsByTagName('xmax')
        xmax = get_xmax[0].firstChild.data
        obj['xmax'] = xmax

        get_ymin = o.getElementsByTagName('ymax')
        ymax = get_ymin[0].firstChild.data
        obj['ymax'] = ymax

        objs.append(obj)

    return img_info, objs
