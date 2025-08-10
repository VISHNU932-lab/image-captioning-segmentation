import streamlit as st
from PIL import Image
import zipfile, io, os
from model_utils import load_segmentation_model, load_caption_model, run_segmentation, overlay_masks, generate_caption

st.set_page_config(page_title='Image Captioning + Segmentation', layout='centered')
st.title('Image Captioning + Segmentation (Streamlit)')

# Load models once
with st.spinner('Loading models (this may take a minute on first run)...'):
    SEG_MODEL = load_segmentation_model(device='cpu')
    CAP_PROC, CAP_MODEL = load_caption_model(device='cpu')

st.write('Upload a single image, or a ZIP of images to process in batch.')

mode = st.radio('Mode', ['Single Image', 'Batch (ZIP)'])

if mode == 'Single Image':
    uploaded = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        masks, labels, scores = run_segmentation(SEG_MODEL, img, score_thresh=0.5)
        seg_viz = overlay_masks(img, masks, alpha=0.4) if masks else img
        caption = generate_caption(CAP_PROC, CAP_MODEL, img)
        st.image(seg_viz, caption=caption, use_column_width=True)
        st.success('Caption: ' + caption)

else:
    zip_file = st.file_uploader('Upload a ZIP of images', type=['zip'])
    if zip_file:
        in_memory = io.BytesIO(zip_file.read())
        with zipfile.ZipFile(in_memory) as z:
            names = [n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))]
            if len(names) == 0:
                st.warning('No images found in ZIP.')
            else:
                out_io = io.BytesIO()
                with zipfile.ZipFile(out_io, mode='w') as out_z:
                    progress = st.progress(0)
                    for i, name in enumerate(names, 1):
                        data = z.read(name)
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                        masks, labels, scores = run_segmentation(SEG_MODEL, img, score_thresh=0.5)
                        seg_viz = overlay_masks(img, masks, alpha=0.4) if masks else img
                        caption = generate_caption(CAP_PROC, CAP_MODEL, img)
                        buf = io.BytesIO()
                        seg_viz.save(buf, format='PNG')
                        out_z.writestr(name + '.overlay.png', buf.getvalue())
                        out_z.writestr(name + '.caption.txt', caption)
                        progress.progress(i/len(names))
                out_io.seek(0)
                st.download_button('Download results ZIP', data=out_io.getvalue(), file_name='results.zip')
