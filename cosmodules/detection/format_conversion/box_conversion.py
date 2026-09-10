from typing import Literal, Tuple


class ConvertBoxFromAnyToVOC:
    def voc(self, xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[int, int, int, int]:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        return xmin, ymin, xmax, ymax

    def coco(self, xmin: int, ymin: int, width: int, height: int) -> Tuple[int, int, int, int]:
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmin + width), int(ymin + height)
        return xmin, ymin, xmax, ymax

    def yolo(
            self,
            cx: float,
            cy: float,
            w: float,
            h: float,
            img_width: int,
            img_height: int
        ) -> Tuple[int, int, int, int]:
        xmin = round((float(cx) - float(w) / 2) * float(img_width))
        ymin = round((float(cy) - float(h) / 2) * float(img_height))
        xmax = round((float(cx) + float(w) / 2) * float(img_width))
        ymax = round((float(cy) + float(h) / 2) * float(img_height))
        return xmin, ymin, xmax, ymax

    def run(
            self,
            src_type: Literal["voc", "yolo", "coco"],
            b1: int | float,
            b2: int | float,
            b3: int | float,
            b4: int | float,
            **kwargs
        ) -> Tuple[int, int, int, int]:
        func = getattr(self, src_type)
        out = func(b1, b2, b3, b4, **kwargs)
        return out


class ConvertBoxFromVOCToAny:
    def voc(self, xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[int, int, int, int]:
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def coco(self, xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[int, int, int, int]:
        xmin = int(xmin)
        ymin = int(ymin)
        width = int(xmax) - int(xmin)
        height = int(ymax) - int(ymin)
        return xmin, ymin, width, height

    def yolo(
            self,
            xmin: int,
            ymin: int,
            xmax: int,
            ymax: int,
            img_width: int,
            img_height: int
        ) -> Tuple[float, float, float, float]:
        cx = round((int(xmin) + int(xmax)) / 2 / float(img_width), 6)
        cy = round((int(ymin) + int(ymax)) / 2 / float(img_height), 6)
        w  = round((int(xmax) - int(xmin)) / float(img_width), 6)
        h  = round((int(ymax) - int(ymin)) / float(img_height), 6)
        return cx, cy, w, h

    def run(
            self,
            xmin: int,
            ymin: int,
            xmax: int,
            ymax: int,
            target_type: Literal["voc", "yolo", "coco"],
            **kwargs
        ) -> Tuple[int, int, int, int]:
        func = getattr(self, target_type)
        out = func(xmin, ymin, xmax, ymax, **kwargs)
        return out
