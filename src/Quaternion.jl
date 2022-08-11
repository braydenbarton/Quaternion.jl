module Quaternions
export ⊗, qconj, qvq, qnorm, qinv

"""
    ⊗(p, q)

Defines the quaternion composition operation for right-handed quaternions of the form [w; x; y; z] where `w` is 
the scalar component and `xyz` are the vector components.
"""
function ⊗(p::Vector, q::Vector)
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    [
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw
    ]
end

"""
    qconj(q)

Calculates the conjugate of the given quaternion.
"""
qconj(q::Vector) = [q[1]; -q[2:4]]

"Shorthand function for rotating a 3-element vector with the given quaternion"
qvq(q::Vector, v::Vector) = @view (q⊗[0; v]⊗qconj(q))[2:4]

"Calculates the norm of a quaternion without using the LinearAlgebra package"
qnorm(q::Vector) = sqrt(q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)

"Calculates the inverse of a quaternion in an efficient fashion"
qinv(q::Vector) = qconj(q) / (q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)

end
