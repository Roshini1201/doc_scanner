import cv2
import numpy as np

# print("OpenCV:", cv2.__version__)
# print("NumPy:", np.__version__)

# IDENTIFY EDGES
img = cv2.imread(r"C:\Users\kbrmu\Desktop\CODE\Project\doc_scanner\cash-receipt.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# USE CANNY EDGE DETECTOR
edges = cv2.Canny(blur, 100, 200)

# FIND CONTOURS 
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# SORT THE CONTOURS BASED ON AREA - DECREASING ORDER
contours = sorted(
    contours,
    key= cv2.contourArea,
    reverse= True
)

# MAIN DOC CONTOUR IS NONE INITIALLY
doc_contr = None

# ITERATE THROUGH EACH CONTOUR IDENTIFIED 
for contour in contours:
    # FIND PERIMETER OF THE CLOSED CONTOUR 
    perimeter = cv2.arcLength(contour, True)

    # IDENTIFY THE VERTICES WITH APPROX TOLERANCE OF 2% OF PERIMETER
    approx = cv2.approxPolyDP(
        contour, 0.02*perimeter, True
    )

    # CHECK IF THERE ARE 4 VERTICES OF THE DOC 
    if len(approx)==4 and cv2.contourArea(approx)>1000:
        # ASSIGN THE BIGGEST CONTOUR APPROX VAL TO DOC CONTOUR 
        doc_contr = approx
        break

# CHECK IF THERE IS A DOC CONTOUR
if doc_contr is not None:

    # DRAW ON IMG WITH DOC CONTR VALS INCLUDING ALL IN GREEN COLOR WITH THICKNESS 3
    borders = img.copy()
    cv2.drawContours(borders, [doc_contr], -1, [0,255,0], 3)
    print("Document Corners: \n", doc_contr.reshape(4,2))

    pts = doc_contr.reshape(4,2).astype(np.float32)
    rect = np.zeros((4, 2), dtype= np.float32)

    # CALC SUM AND DIFF BETWEEN CONSEQUTIVE ELEMENTS AND FLATTEN TO MAKE IT 1D
    s = pts.sum(axis= 1)
    d = np.diff(pts, axis= 1).flatten()

    # FIND THE COORDINATES
    # TOP LEFT CORNER USUALLY HAS SMALLEST SUM AND BOTTOM RIGHT HAS LARGEST SUM VAL
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # USUALLY TOP RIGHT CORNER HAS SMALLEST DIFF AND BOTTOM LEFT HAS LARGEST DIFF
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    dest = np.float32([
        [0,0],[300,0],
        [300,500],[0,500]
    ])

    # PERSPECTIVE TRANSFORMATION 
    matrix = cv2.getPerspectiveTransform(rect, dest)
    wrapped = cv2.warpPerspective(img, matrix, (300,500))

    # DISPLAY THE DOC
    cv2.imshow("Document Edges", edges)
    cv2.imshow("Document Borders", borders)
    cv2.imshow("Scanned Documents", wrapped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Document not found")
