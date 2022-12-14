import numpy as np
def my_cholesky(matrix, precision=np.finfo(float).eps, fix_type='replace'):
    # Positive-definite via eigvals
    my_chol=np.zeros(matrix.shape)
    if fix_type=='replace':
        matrix[np.any(np.linalg.eigvals(matrix)<=0., axis=-1)]=precision*np.eye(matrix.shape[-1])
    elif fix_type=='add':
        matrix[np.any(np.linalg.eigvals(matrix)<=0., axis=-1)]+=precision*np.eye(matrix.shape[-1])
    # Actual cholesky
    try:
        my_chol=np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in str(err):
            if fix_type=='replace':
                my_chol=np.linalg.cholesky(precision*np.ones(matrix.shape)*np.eye(matrix.shape[-1]))
            elif fix_type=='add':
                my_chol=np.linalg.cholesky(matrix+precision*np.ones(matrix.shape)*np.eye(matrix.shape[-1]))
            elif 'strict' in fix_type:
                for this_matrix_idx in np.ndindex(matrix.shape[:-2]):
                    my_chol[this_matrix_idx]=my_cholesky(matrix[this_matrix_idx], precision, fix_type.split('_')[1])
        else:
            raise
    
    # Return        
    return my_chol
