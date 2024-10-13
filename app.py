# Paste the entire Streamlit app code here (the code from the previous artifact)

# Run the Streamlit app
!streamlit run app.py &>/dev/null&

# Get the public URL
from google.colab import output
output.serve_kernel_port_as_window(8501)
