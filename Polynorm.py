import torch

class XIELUPolyNorm(nn.Module):
    """
    XIELUPolyNorm activation with Taylor polynomial patch and official PolyNorm implementation.
    
    Combines XIELU base function (with Taylor patch for numerical stability) 
    and official PolyNorm transformation for enhanced representation power.
    
    Official PolyNorm: weight[0] * norm(x³) + weight[1] * norm(x²) + weight[2] * norm(x) + bias
    """
    
    def __init__(self, 
                 alpha_p_init: float = 0.8,
                 alpha_n_init: float = 0.8, 
                 beta: float = 0.5, 
                 eps: float = 1e-4,
                 epsilon: float = 1e-6):
        """
        Initialize XIELUPolyNorm with Taylor patch and official PolyNorm.
        
        Args:
            alpha_p_init: Initial value for positive branch quadratic coefficient
            alpha_n_init: Initial value for negative branch exponential coefficient  
            beta: Linear coefficient (controls derivative at x=0)
            eps: Width of Taylor patch region [-ε, 0] for XIELU
            epsilon: Numerical stability parameter for PolyNorm normalization
        """
        super().__init__()
        
        # XIELU parameters (same as before)
        self.beta = beta
        self.eps = eps
        
        # Taylor coefficients: φ(x) = x + x²/2 + x³/6
        self.register_buffer('taylor_a', torch.tensor(1.0/6.0))  # x³
        self.register_buffer('taylor_b', torch.tensor(1.0/2.0))  # x²
        
        # Learnable parameters (softplus parameterization)
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init)) - 1))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta)) - 1))
        
        # Official PolyNorm parameters
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(3) / 3)  # [w₀, w₁, w₂] for [x³, x², x]
        self.bias = nn.Parameter(torch.zeros(1))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Official PolyNorm normalization function.
        Normalizes x using RMS normalization: x * rsqrt(mean(x²) + ε)
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: XIELU base with Taylor patch + official PolyNorm transformation.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Activated tensor of same shape as input
        """
        # Get effective XIELU parameters
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        
        # === XIELU with Taylor patch ===
        u = torch.where(
            x > 0,
            # Positive branch
            alpha_p * x * x + self.beta * x,
            # Negative branch with Taylor patch
            torch.where(
                x >= -self.eps,
                # Taylor patch: φ(x) = x + x²/2 + x³/6
                alpha_n * (x + self.taylor_b * x * x + self.taylor_a * x * x * x) - alpha_n * x + self.beta * x,
                # Original exponential
                alpha_n * torch.expm1(x) - alpha_n * x + self.beta * x
            )
        )
        
        # === Official PolyNorm transformation ===
        # Compute polynomial terms
        u1 = u           # x¹
        u2 = u * u       # x²
        u3 = u2 * u      # x³
        
        # Apply normalization to each term
        norm_u3 = self._norm(u3)  # Normalized x³
        norm_u2 = self._norm(u2)  # Normalized x²
        norm_u1 = self._norm(u1)  # Normalized x¹
        
        # Weighted combination: w₀*norm(x³) + w₁*norm(x²) + w₂*norm(x) + bias
        output = (self.weight[0] * norm_u3 + 
                 self.weight[1] * norm_u2 + 
                 self.weight[2] * norm_u1 + 
                 self.bias)
        
        return output



class XIELUPoly(nn.Module):
    """
    XIELUPolyType1 implementing PolyCom Type I with Horner's method (order=3 optimized).
    
    Type I: p = a₀ + a₁·u + a₂·u² + a₃·u³ where u = XIELU(x)
    
    Uses Horner's method for efficient polynomial evaluation:
    P(u) = a₀ + u(a₁ + u(a₂ + u×a₃))
    
    Optimized specifically for order=3 (cubic polynomial).
    """
    
    def __init__(self, 
                 alpha_p_init: float = 0.8,
                 alpha_n_init: float = 0.8, 
                 beta: float = 0.5, 
                 eps: float = 1e-4):
        """
        Initialize XIELUPolyType1 with Horner's method optimization for order=3.
        
        Args:
            alpha_p_init: Initial value for positive branch quadratic coefficient
            alpha_n_init: Initial value for negative branch exponential coefficient  
            beta: Linear coefficient (controls derivative at x=0)
            eps: Width of Taylor patch region [-ε, 0] for XIELU
        """
        super().__init__()
        
        # XIELU parameters
        self.beta = beta
        self.eps = eps
        
        # Fixed order=3 for optimal Horner implementation
        self.order = 3
        
        # Taylor coefficients: φ(x) = x + x²/2 + x³/6
        self.register_buffer('taylor_a', torch.tensor(1.0/6.0))  # x³
        self.register_buffer('taylor_b', torch.tensor(1.0/2.0))  # x²
        
        # Learnable parameters (softplus parameterization)
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init)) - 1))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta)) - 1))
        
        # PolyCom Type I coefficients: [a₀, a₁, a₂, a₃] for [1, u, u², u³]
        # Initialize as in document: a₀ = 0, aₖ = 1/r for k > 0
        self.a0 = nn.Parameter(torch.tensor(0.0))           # Constant term
        self.a1 = nn.Parameter(torch.tensor(1.0/3.0))       # Linear term  
        self.a2 = nn.Parameter(torch.tensor(1.0/3.0))       # Quadratic term
        self.a3 = nn.Parameter(torch.tensor(1.0/3.0))       # Cubic term
    
    def _xielu(self, x: torch.Tensor) -> torch.Tensor:
        """
        XIELU activation with Taylor patch for numerical stability.
        
        Args:
            x: Input tensor
            
        Returns:
            XIELU activated tensor
        """
        # Get effective XIELU parameters
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        
        # === XIELU with Taylor patch ===
        u = torch.where(
            x > 0,
            # Positive branch: α_p * x² + β * x
            alpha_p * x * x + self.beta * x,
            # Negative branch with Taylor patch
            torch.where(
                x >= -self.eps,
                # Taylor patch: φ(x) = x + x²/2 + x³/6
                alpha_n * (x + self.taylor_b * x * x + self.taylor_a * x * x * x) - alpha_n * x + self.beta * x,
                # Original exponential: α_n * (e^x - 1) - α_n * x + β * x
                alpha_n * torch.expm1(x) - alpha_n * x + self.beta * x
            )
        )
        
        return u
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Type I PolyCom using optimized Horner's method for order=3.
        
        P(u) = a₀ + a₁u + a₂u² + a₃u³ = a₀ + u(a₁ + u(a₂ + u×a₃))
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Polynomial composition in activation space
        """
        # Step 1: Apply XIELU activation once
        u = self._xielu(x)
        
        # Step 2: Horner's method evaluation (order=3 specialized)
        # P(u) = a₀ + u(a₁ + u(a₂ + u×a₃))
        result = self.a3                        # Start with a₃
        result = self.a2 + u * result           # a₂ + u×a₃  
        result = self.a1 + u * result           # a₁ + u(a₂ + u×a₃)
        result = self.a0 + u * result           # a₀ + u(a₁ + u(a₂ + u×a₃))
        
        return result
