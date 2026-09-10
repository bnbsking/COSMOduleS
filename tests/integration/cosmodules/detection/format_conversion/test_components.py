from cosmodules.detection.format_conversion.components import (
    VOCComponents
)


class TestVOCComponents:
    def test_all(self):
        voc = VOCComponents
        out = ""
        out += voc.get_xml(
            filename="test.jpg",
            path="example/test.jpg",
            width=800,
            height=600
        )
        out += voc.get_obj(
            name="dog",
            xmin=100,
            ymin=150,
            xmax=400,
            ymax=350
        )
        out += voc.get_end()
        assert out.strip() ==\
        """
<annotation>
    <folder>folder</folder>
    <filename>test.jpg</filename>
    <path>example/test.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>800</width>
        <height>600</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>dog</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>400</xmax>
            <ymax>350</ymax>
        </bndbox>
    </object>
</annotation>
        """.strip()


if __name__ == "__main__":
    obj = TestVOCComponents()
    obj.test_all()
