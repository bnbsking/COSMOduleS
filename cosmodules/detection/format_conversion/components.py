class VOCComponents:
    @staticmethod
    def get_xml(filename: str, path: str, width: int, height: int) -> str:
        return f"\
<annotation>\n\
    <folder>folder</folder>\n\
    <filename>{filename}</filename>\n\
    <path>{path}</path>\n\
    <source>\n\
        <database>Unknown</database>\n\
    </source>\n\
    <size>\n\
        <width>{width}</width>\n\
        <height>{height}</height>\n\
        <depth>3</depth>\n\
    </size>\n\
    <segmented>0</segmented>\n"
    
    @staticmethod
    def get_obj(name: str, xmin: int, ymin: int, xmax: int, ymax: int) -> str:
        return f"\
    <object>\n\
        <name>{name}</name>\n\
        <pose>Unspecified</pose>\n\
        <truncated>0</truncated>\n\
        <difficult>0</difficult>\n\
        <bndbox>\n\
            <xmin>{xmin}</xmin>\n\
            <ymin>{ymin}</ymin>\n\
            <xmax>{xmax}</xmax>\n\
            <ymax>{ymax}</ymax>\n\
        </bndbox>\n\
    </object>\n"

    @staticmethod
    def get_end() -> str:
        return "</annotation>"
    