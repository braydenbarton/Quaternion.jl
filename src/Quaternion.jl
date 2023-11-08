module Quaternion
using LinearAlgebra, StaticArrays
export ⊗, qcomp, qconj, qvq, qnorm, qinv, axis_angle_to_q, q_to_axis_angle, q2mat, axis_angle_to_mat, mat_to_axis_angle, 
    mat2q

# Declaring convenience type aliases
"Abstract quaternion representation"
const Quat{T} = StaticVector{4, T} where T<:Real
quat(q::AbstractVector) = SizedVector{4}(q)
quat(q::Quat) = q
"Abstract representation for real 3-vecs (position, velocity, angular velocity, etc)"
const R3Vec{T} = StaticVector{3, T} where T<:Real
r3vec(vec::AbstractVector) = SizedVector{3}(vec)
r3vec(vec::R3Vec) = vec

"""
    ⊗(p, q)
    qcomp(p, q)

Defines the quaternion composition operation for right-handed quaternions of the form [w; x; y; z] where `w` is 
the scalar component and `xyz` are the vector components.
"""
function qcomp(p::Quat, q::Quat)
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    SA[
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw
    ]
end
⊗(p, q) = qcomp(p, q)
"AbstractVector wrapping for old code"
qcomp(p::AbstractVector, q::AbstractVector) = (⊗(quat(p), quat(q)))

"""
    qconj(q)

Calculates the conjugate of the given quaternion.
"""
qconj(q::Quat) = [q[1]; -q[SA[2:4...]]]
qconj(q::AbstractVector) = qconj(quat(q))

"Shorthand function for rotating a 3-element vector with the given quaternion"
qvq(q::Quat, v::R3Vec) = (q ⊗ [0; v] ⊗ qconj(q))[SA[2:4...]]
qvq(q::AbstractVector, v::AbstractVector) = qvq(quat(q), r3vec(v))

"Calculates the norm of a quaternion without using the LinearAlgebra package"
qnorm(q::Quat) = sqrt(q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)
qnorm(q::AbstractVector) = qnorm(quat(q))

"Calculates the inverse of a quaternion in an efficient fashion"
qinv(q::Quat) = qconj(q) / (q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)
qinv(q::AbstractVector) = qinv(quat(q))

"Converts an axis of rotation and an angle of rotation (in radians) to a quaternion"
axis_angle_to_q(vec::R3Vec, θ::Real) = [cos(θ/2); vec/norm(vec)*sin(θ/2)]
axis_angle_to_q(vec::AbstractVector, θ::Real) = axis_angle_to_q(r3vec(vec), θ)

"Recovers the axis-angle representation from a quaternion"
q_to_axis_angle(q::R3Vec) = @views (q[2:4]/norm(q[2:4]), 2*atan(norm(q[2:4]), q[1]))
q_to_axis_angle(q::AbstractVector) = q_to_axis_angle(quat(q))

"Converts a rotation quaternion to a rotation matrix"
function q2mat(q::Quat)
    s = qnorm(q) ^ -2
    qw, qx, qy, qz = q
    SA[
        1 - 2s*(qy^2 + qz^2)    2s*(qx*qy - qz*qw)      2s*(qx*qz + qy*qw);
        2s*(qx*qy + qz*qw)      1 - 2s*(qx^2 + qz^2)    2s*(qy*qz - qx*qw);
        2s*(qx*qz - qy*qw)      2s*(qy*qz + qx*qw)      1 - 2s*(qx^2 + qy^2)
    ]
end
q2mat(q::AbstractVector) = q2mat(quat(q))

"Converts an axis of rotation and an angle of rotation (in radians) to a rotation matrix"
axis_angle_to_mat(vec::R3Vec, θ::Real) = q2mat(axis_angle_to_q(vec, θ))

"Recovers the axis-angle representation from a rotation matrix"
function mat_to_axis_angle(R::AbstractMatrix)
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
mat2q(R::AbstractMatrix) = axis_angle_to_q(mat_to_axis_angle(R)...)

end
