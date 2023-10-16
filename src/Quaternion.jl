module Quaternion
using LinearAlgebra
export ⊗, qconj, qvq, qnorm, qinv, axis_angle_to_q, q_to_axis_angle, q2mat, axis_angle_to_mat, mat_to_axis_angle, mat2q

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
qvq(q::Vector, v::Vector) = (q⊗[0; v]⊗qconj(q))[2:4]

"Calculates the norm of a quaternion without using the LinearAlgebra package"
qnorm(q::Vector) = sqrt(q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)

"Calculates the inverse of a quaternion in an efficient fashion"
qinv(q::Vector) = qconj(q) / (q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)

"Converts an axis of rotation and an angle of rotation (in radians) to a quaternion"
axis_angle_to_q(vec::Vector, θ::Real) = [cos(θ/2); vec/norm(vec)*sin(θ/2)]

"Recovers the axis-angle representation from a quaternion"
q_to_axis_angle(q::Vector) = @views (q[2:4]/norm(q[2:4]), 2*atan(norm(q[2:4]), q[1]))

"Converts a rotation quaternion to a rotation matrix"
function q2mat(q::Vector)
    s = qnorm(q) ^ -2
    qw, qx, qy, qz = q
    [
        1 - 2s*(qy^2 + qz^2)    2s*(qx*qy - qz*qw)      2s*(qx*qz + qy*qw);
        2s*(qx*qy + qz*qw)      1 - 2s*(qx^2 + qz^2)    2s*(qy*qz - qx*qw);
        2s*(qx*qz - qy*qw)      2s*(qy*qz + qx*qw)      1 - 2s*(qx^2 + qy^2)
    ]
end

"Converts an axis of rotation and an angle of rotation (in radians) to a rotation matrix"
axis_angle_to_mat(vec::Vector, θ::Real) = q2mat(axis_angle_to_q(vec, θ))

"Recovers the axis-angle representation from a rotation matrix"
function mat_to_axis_angle(R::Matrix)
    vec = nullspace(R-I)[:, 1]
    vec /= norm(vec)
    θ = acos((sum(diag(R)) - 1) / 2)
    
    # Find a vector orthogonal to vec
    if abs(vec[1]) > abs(vec[2])
        vec_a = [0, 1, 0]
    else
        vec_a = [1, 0, 0]
    end
    vec_b = cross(vec, vec_a)

    # Resolve sign ambiguity in θ
    if vec ⋅ cross(vec_b, R*vec_b) < 0
        θ *= -1
    end

    return vec, θ
end

"Converts a rotation matrix to a quaternion"
mat2q(R::Matrix) = axis_angle_to_q(mat_to_axis_angle(R)...)

end
