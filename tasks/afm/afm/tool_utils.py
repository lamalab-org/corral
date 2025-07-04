from corral.base import Tool, ToolArgument
from corral.utils import tool
from typing import Dict
from dotenv import load_dotenv
import os
import modal
from loguru import logger
import subprocess
import time
import nanosurf

import getpass
import os
import functools
import operator
import glob
import nanosurf
import time
import numpy as np
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import tool
from NSFopen.read import read
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.optimize import curve_fit
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from typing import Sequence, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

db_new = Chroma(persist_directory="C:/Users/Admin/Desktop/corral/mat-agent-bench/tasks/afm/afm/AILA/AILA-main/app/aila_db", embedding_function=embeddings)

retriever_wo = db_new.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


Document_Retriever = create_retriever_tool(
    retriever_wo,
    "Document_Retriever",
    "This tool offers reference code specifically designed for "
    "managing an AFM (Atomic Force Microscope) machine, which is connected to a database. "
    "The tool also includes the AFM software manual for guidance."
    "However, it does not contain any code related to displaying/optimizing images."
    "Single query allowed at one. but multiple call allowed",
)
# @tool
# def Document_Retriever(query: str) -> str:
#     """
#     This tool retrieves code snippets from a database that are specifically designed for
#     operating an Atomic Force Microscope (AFM) machine. The retrieved code is intended to be
#     used as a reference for controlling the AFM.

#     Args:
#         query (str): The query string to search for relevant code snippets in the database.

#     Returns:
#         str: The retrieved code snippet.
#     """
#     result = Document_Retriever.invoke(query)
#     return result


def scan_image(PGain, IGain, DGain, file):
    spm = nanosurf.SPM()
    application = spm.application
    scan = application.Scan
    zcontrol = application.ZController
    
    application.SetGalleryHistoryFilenameMask(file)
    zcontrol.PGain = PGain
    zcontrol.IGain = IGain
    zcontrol.DGain = DGain
    scan.StartFrameUp()
    
    scanning = scan.IsScanning
    while scanning:
        print("Scanning in progress...")
        time.sleep(5)
        scanning = scan.IsScanning

    from NSFopen.read import read
    list_of_files = glob.glob(new_path+'/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    afm = read(latest_file)
    data = afm.data
    im_file_fw = data['Image']['Forward']['Z-Axis']
    im_file_bw = data['Image']['Backward']['Z-Axis']
    similarity_index, diff = ssim(im_file_bw, im_file_fw, full=True, data_range=im_file_bw.max() - im_file_bw.min())
    mse = mean_squared_error(im_file_bw, im_file_fw)
    del spm
    return similarity_index, mse

def corrected_image(image):
    def poly5d(xy, *params):
        x, y = xy
        return (params[0] + params[1]*x + params[2]*y + 
                params[3]*x**2 + params[4]*y**2 + 
                params[5]*x*y + params[6]*x**3 + params[7]*y**3 +
                params[8]*x**2*y + params[9]*x*y**2 + 
                params[10]*x**4 + params[11]*y**4 + 
                params[12]*x**3*y + params[13]*x*y**3 +
                params[14]*x**2*y**2)

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    image_flat = image.flatten()
    params, _ = curve_fit(poly5d, (x, y), image_flat, p0=np.zeros(15))
    baseline = poly5d((x, y), *params).reshape(image.shape)
    corrected_image = image - baseline
    return corrected_image

def scan_image_poly(PGain, IGain, DGain, file):
    spm = nanosurf.SPM()
    application = spm.application
    scan = application.Scan
    zcontrol = application.ZController
    
    application.SetGalleryHistoryFilenameMask(file)
    zcontrol.PGain = PGain
    zcontrol.IGain = IGain
    zcontrol.DGain = DGain
    scan.StartFrameUp()
    
    scanning = scan.IsScanning
    while scanning:
        print("Scanning in progress...")
        time.sleep(5)
        scanning = scan.IsScanning

    from NSFopen.read import read
    list_of_files = glob.glob(new_path+'/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    afm = read(latest_file)
    data = afm.data
    im_file_fw = corrected_image(data['Image']['Forward']['Z-Axis'])
    im_file_bw = corrected_image(data['Image']['Backward']['Z-Axis'])
    similarity_index, diff = ssim(im_file_bw, im_file_fw, full=True, data_range=im_file_bw.max() - im_file_bw.min())
    mse = mean_squared_error(im_file_bw, im_file_fw)
    del spm
    return similarity_index, mse

class MyProblem(ElementwiseProblem):
    def __init__(self, baseline=True):
        super().__init__(n_var=3,
                         n_obj=1,
                         xl=np.array([0, 500, 0]),
                         xu=np.array([500, 9000, 100]))
        self.baseline = baseline

    def _evaluate(self, x, out, *args, **kwargs):
        if self.baseline:
            scan_outputs = scan_image_poly(x[0], x[1], x[2], f"scan_{x[0]}_{x[1]}_{x[2]}_")
        else:
            scan_outputs = scan_image(x[0], x[1], x[2], f"scan_{x[0]}_{x[1]}_{x[2]}_")

        mse = scan_outputs[1]
        f1 = (1 - mse) * 10000
        out["F"] = [f1]