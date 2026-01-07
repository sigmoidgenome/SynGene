import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from random import randint
import io        # For handling memory streams
import zipfile   # For creating the ZIP file

st.title("Synthetic Gene Expression Generator")

st.write("Welcome")

st.write("Specify the number of gene expression clusters, the number of genes in each cluster, and the number of timepoints per gene expression profile. The number of timepoints corresponds to how frequently expression levels are measured—higher values produce smoother expression curves.")
Clusters = st.slider("Clusters: ", 0, 10, 1)
gene_required = st.slider("Number of genes in each cluster: ", 0, 1000, 10)
Length = st.slider("Timepoints: ", 0, 1000, 10)

if st.button("Generate"):
    zip_buffer= io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
    
        Scaler = MinMaxScaler(feature_range=(-1, 1))
        
        for cluster in range(Clusters): 
            A = randint(10, 100)
            print(f"Amplitude = {A}")
            Freq = randint(5, 50)
            print(f"Frequency = {Freq}")
            phase = np.random.uniform(0, 2 *np.pi ) 
            Time = np.linspace(0, Length/1000, Length) #we have defined x as a time series above and will be using it here. x = np.linspace(0, 0.1, 100)
            sinewave = A * np.sin(Freq * Time + phase)
        
            A2 = randint(10, 100)
            Freq2 = randint(5, 50)
            phase2 = np.random.uniform(0, 2 *np.pi )
            coswave = A2 * np.cos(Freq2*Time + phase2)
        
            y = np.convolve(sinewave, coswave, mode = "same")
            y_scaled = Scaler.fit_transform(y.reshape(-1,1))
            y_scaled_reshape = y_scaled.reshape(-1)+0.1  
        
            matrix = np.dot(y_scaled, y_scaled.T)
            normalising_value = np.dot(y_scaled_reshape.T, y_scaled_reshape)
            cov_matrix = matrix/normalising_value
        
            for m in range(cov_matrix.shape[0]): #iterating through each row
              for n in range(cov_matrix.shape[1]):
                if abs(m-n)>50:
                  cov_matrix[m,n]=0
                  
            gene = []
            for i in range(gene_required):
              i = np.random.multivariate_normal(mean=y_scaled_reshape, cov = cov_matrix)
              gene.append(i)
        
            gene = np.asarray(gene)
        
            noise = np.random.normal(loc=0, scale=0.05, size=gene.shape)
            gene_noisy = gene + noise
        
            fig, ax = plt.subplots(figsize=(10, 5))
        
            for m in range(gene_noisy.shape[0]):
              ax.plot(Time, gene_noisy[m,:], alpha=0.6) #label = f"gene = {m}"
        
            ax.plot(Time, y_scaled, color="black", alpha = 1, linewidth = 2.5, label = "Cluster Mean")
            ax.set_title(f"Synthetic gene expression from Cluster {cluster}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Expression")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            #plt.savefig(f"Cluster{cluster}.png")
            
            csv_data = pd.DataFrame(gene_noisy).to_csv(index=False)
            zf.writestr(f"cluster_{cluster}.csv", csv_data)
            
            # Save Image to Zip
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png')
            zf.writestr(f"cluster_{cluster}.png", img_buffer.getvalue())
            
            plt.close(fig)


    zip_buffer.seek(0)
    # Save the zip file to session state so it persists
    st.session_state['zip_data'] = zip_buffer

if st.session_state['zip_data'] is not None:
    st.download_button(
        label="Download All Results (.zip)",
        data=st.session_state['zip_data'],
        file_name="gene_expression_results.zip",
        mime="application/zip"
    )