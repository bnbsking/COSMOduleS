from cosmodules.detection.visualization.show import show_general


def test_show_general():
    img_path = "/app/scripts/layout_example.png"
    ant_path = "/app/scripts/general.json"
    save_path = "/app/scripts/layout_example_gt.jpg"
    show_general(img_path, ant_path, save_path)


def test_show_general_pd():
    img_path = "/app/scripts/test_run_0.jpg"
    ant_path = "/app/scripts/general_nanya.json"
    save_path = "/app/scripts/nanya_pd.jpg"
    show_general(img_path, ant_path, save_path)


if __name__ == "__main__":
    #test_show_general()
    test_show_general_pd()