import numpy as np

def find_mostfre(array:np.ndarray):
    '''find the most frequent element

    Args:
        array: an array

    Returns:
        e : the most frequent element
    '''

    unique_elements, counts = np.unique(array, return_counts=True)
    e = unique_elements[np.argmax(counts)]

    return e


class TreeNode(object):
    """Tree class
    
    Attributes:
        left: node of the left child (with the cut feature <= cut value)
        right: node of the right child (with the cut feature > cut value)
        parent: parent of the current node
        cutoff_id: id of the cut feature
        cutoff_val: cut-off value
        prediction: prediction of the node, i.e., weights * Y
    """
    
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        """initialize the class

        Args:
            left: node of the left child (with the cut feature <= cut value)
            right: node of the right child (with the cut feature > cut value)
            parent: parent of the current node
            cutoff_id: id of the cut feature
            cutoff_val: cut-off value
            prediction: prediction of the node, i.e., weights * Y  
        """
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction


def sqsplit(xTr,yTr,weights=None):
    '''find the cut feature id and the cut value
    
    Args:
        xTr: (n, d) matrix of data points
        yTr: n-dimensional vector of labels
        weights: n-dimensional weight vector for data points
    
    Returns:
        feature: index of the best cut's feature, None indicates splitting cannot be done
        cut: cut-value of the best cut,  None indicates splitting cannot be done
    '''
    num, dim = xTr.shape
    if num <= 1:
        raise ValueError('xTr must have at least two samples')
    if dim == 0:
        raise ValueError('xTr must have at least one feature')

    if weights is None: # if no weights are passed on, assign uniform weights
        weights = np.ones(num)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)

    bestloss = np.inf
    feature = None # return None when the features are the same 
    cut = None # return None when the features are the same

    # traverse each feature dimension
    for j in range(dim): 
        features = np.array(xTr[:,j])
        sorted_indices = np.argsort(features)
        w_left = 0; w_right = np.sum(weights)
        p_left = 0; p_right = np.dot(weights, yTr)
        q_left = 0; q_right = np.dot(weights, yTr ** 2)

        # traverse the feature of each sample in the asc order
        for ii in range(num-1):
            i = sorted_indices[ii]
            i_1 = sorted_indices[ii+1]
            # updata W, P, Q
            w_left += weights[i]; w_right -= weights[i]
            p_left += weights[i] * yTr[i]; p_right -= weights[i] * yTr[i]
            q_left += weights[i] * yTr[i]**2; q_right -= weights[i] * yTr[i]**2
            # manage the situation where the two samples's features are the same
            if features[i] == features[i_1]:
                continue              
            loss_new = q_left - p_left**2/w_left + q_right - p_right**2/w_right
            if loss_new < bestloss:
                bestloss = loss_new
                feature = j
                cut = (features[i] + features[i_1])/2

    return feature, cut


def cart(xTr,yTr,max_depth=np.inf,weights=None,pred_type='most-frequent'):
    '''build a CART tree using a queue
    
    The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).

    Args:
        xTr: (n,d) matrix of data
        yTr: n-dimensional vector
        max_depth: maximum tree depth
        weights: n-dimensional weight vector for data points

    Returns: 
        tree: root of decision tree
    '''
    if pred_type == 'most-frequent':
        pred_funct = lambda y,w : find_mostfre(y)
    elif pred_type == 'avg':
        pred_funct = lambda y,w : np.dot(y,w)
    else:
        raise ValueError(f'Unknown pred_type {pred_type}')

    num = xTr.shape[0]
    if weights is None:
        w = np.ones(num) / num
    else:
        w = weights

    if num == 1:
        return TreeNode(None,None,None,None,None, pred_funct(yTr, w))
    
    feature, cut = sqsplit(xTr,yTr,w)
    root = TreeNode(None,None,None,feature,cut, None)
    queue = [[root,xTr,yTr,w,1]] # store information: [TreeNode, samples in the node, labels, depth]

    while queue != []:
        node, xTr, yTr, w, current_depth= queue.pop(0)
        # skip when the node is a leaf
        # in this case, the leaf is the node when features of the samples are the same
        if node.cutoff_id is None:
            node.prediction = pred_funct(yTr, w)
            continue
        # make the node a leaf and skip when meeting the max depth
        if current_depth == max_depth:
            node.cutoff_id = None
            node.cutoff_val = None
            node.prediction = pred_funct(yTr, w)
            continue

        current_depth += 1
        # split the node
        left_bool = xTr[:,node.cutoff_id] <= node.cutoff_val            
        right_bool = ~left_bool
        # check if left node is a leaf
        if np.sum(left_bool) == 1:
            node.left = TreeNode(None,None,node,None,None, yTr[left_bool][0])
        else:   
            feature, cut = sqsplit(xTr[left_bool,:],yTr[left_bool],w[left_bool])
            node_left = TreeNode(None,None,node,feature,cut,None)
            node.left = node_left
            # add the left node to the queue
            queue.append([node_left,xTr[left_bool],yTr[left_bool],w[left_bool] ,current_depth])
        # check if right node is a leaf
        if np.sum(right_bool) == 1:
            node.right = TreeNode(None,None,node,None,None, yTr[right_bool][0])
        else:
            feature, cut = sqsplit(xTr[right_bool,:],yTr[right_bool],w[right_bool])
            node_right = TreeNode(None,None,node,feature,cut, None)
            node.right = node_right
            # add the right node to the queue
            queue.append([node_right,xTr[right_bool],yTr[right_bool],w[right_bool] ,current_depth])
        
    return root

def pred_tree(root, xTe):
    '''get the predictions of xTe according to the trained decision tree
    
    Input:
        root: TreeNode of the decision tree
        xTe:  (n, d) matrix of data points

    Output:
        pred: n-dimensional vector of predictions, 0-1 for classification, continuous values for regression
    '''
    if root is None:
        raise ValueError('the tree node is None')
    
    num = xTe.shape[0]
    idx = np.arange(num)
    pred = np.zeros(num)
    queue = [(root,idx)]

    while queue != []:
        node,idx = queue.pop(0)
        # check if the node is a leaf
        if node.cutoff_id is None:
            pred[idx] = node.prediction
        else: 
            # go to the left node or right node respectively
            left_bool = xTe[idx,node.cutoff_id] <= node.cutoff_val
            right_bool = ~left_bool
            left_indices = idx[left_bool]
            right_indices = idx[right_bool]
            if len(left_indices) > 0:
                queue.append((node.left,left_indices))
            if len(right_indices) > 0:
                queue.append((node.right,right_indices))

    return pred


def print_tree(node, depth=0, prefix="Root"):
    ''' Recursively prints the tree structure. 
    
    Input:
        node: the current TreeNode
        depth: the current depth
        prefix: the prefix to print before the node information
    '''
    if node is not None:
        print(" " * depth * 2 + prefix + ": [Feature: {}, Cut: {}, Prediction: {}]".format(node.cutoff_id, node.cutoff_val, node.prediction))
        print_tree(node.left, depth + 1, "L")
        print_tree(node.right, depth + 1, "R")


def forest(xTr, yTr, m, maxdepth=np.inf):
    '''create a random forest.
    
    Input:
        xTr: (n, d) matrix of data points
        yTr: n-dimensional vector of labels
        m: number of trees in the forest
        maxdepth: maximum depth of tree
        
    Output:
        trees: list of TreeNode decision trees of length m
    '''
    
    num = xTr.shape[0]
    trees = []

    for i in range(m):
        indices = np.random.choice(num, num)
        X = xTr[indices,:]
        Y = yTr[indices]
        tree = cart(X,Y,maxdepth)
        trees.append(tree)
        
    return trees

def pred_forest(trees, X, alphas = None):
    '''get the predictions according to the trained forest.
    
    Input:
        trees: list of TreeNode decision trees of length m
        X: (n,d) matrix of data points
        alphas: m-dimensional weight vector
        cls: True indicates classification
        
    Output:
        pred: n-dimensional vector of predictions
    '''
    m = len(trees)
    if alphas is None:
        alphas = np.ones(m) / len(m)
    preds = []

    for tree in trees:
        pred = pred_tree(tree,X)
        preds.append(pred)
    preds = np.array(preds)
    pred = np.dot(alphas, preds)
        
    return pred

def ada_boost(x,y,maxiter=100,maxdepth=2):
    '''create a boosted decision tree.
    
    Input:
        x: (n, d) matrix of data points
        y:  n-dimensional vector of labels
        maxiter: maximum number of trees
        maxdepth: maximum depth of a tree
        
    Output:
        forest: list of TreeNode decision trees of length m
        alphas: m-dimensional weight vector
        
    (note, m is at most maxiter, but may be smaller,
    as dictated by the Adaboost algorithm)
    '''
    labels = np.unique(y)
    if len(labels) > 2:
        raise ValueError('must be classification while the inputs have {} labels'.format(len(labels)))
    
    avg = np.mean(labels)
    # convert labels to -1, 1
    y[y>=avg] = 1 # '=' manages the situation whese the labels are the same
    y[y<avg] = -1

    num = x.shape[0]
    weights = np.ones(num) / num
    forest  = []
    alphas  = []

    for i in range(maxiter):
        tree = cart(x,y,maxdepth,weights)
        pred = pred_tree(tree,x)
        error = np.sum(weights * (pred-y)**2)
        alpha = np.log((1-error)/error)
        weights = weights * np.exp(alpha * np.abs(pred - y))
        weights = weights / np.sum(weights)
        forest.append(tree)
        alpha.append(alphas)
    
    return forest, alphas