from sklearn.datasets import fetch_olivetti_faces as load_faces

faces = load_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)
print("Keys:", faces.keys()) # display keys
print("Total samples and image size:", faces.images.shape)
print("Total samples and features:", faces.data.shape)
print("Total samples and targets:", faces.target.shape)