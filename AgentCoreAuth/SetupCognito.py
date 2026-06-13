import boto3
import getpass

# Prompt user for inputs
user_pool_name = input("Enter User Pool Name: ")
client_name = input("Enter App Client Name: ")
username = input("Enter Username: ")
permanent_password = getpass.getpass("Enter Permanent Password (will be hidden): ")

# Constants
region = 'us-east-1'
temporary_password = "TempPass123@#!"  # Must meet the password policy

# Create the Cognito IDP client
cognito = boto3.client('cognito-idp', region_name=region)

# Step 1: Create User Pool
print("Creating user pool...")
pool_response = cognito.create_user_pool(
    PoolName=user_pool_name,
    Policies={
        'PasswordPolicy': {
            'MinimumLength': 8
        }
    }
)
pool_id = pool_response['UserPool']['Id']
print(f"User Pool ID: {pool_id}")

# Step 2: Create App Client
print("Creating app client...")
client_response = cognito.create_user_pool_client(
    UserPoolId=pool_id,
    ClientName=client_name,
    GenerateSecret=False,
    ExplicitAuthFlows=['ALLOW_USER_PASSWORD_AUTH', 'ALLOW_REFRESH_TOKEN_AUTH']
)
client_id = client_response['UserPoolClient']['ClientId']
print(f"App Client ID: {client_id}")

# Step 3: Create user with a temporary password (suppress email)
print("Creating user with temporary password...")
cognito.admin_create_user(
    UserPoolId=pool_id,
    Username=username,
    TemporaryPassword=temporary_password,
    MessageAction='SUPPRESS'
)

# Step 4: Set permanent password
print("Setting permanent password...")
cognito.admin_set_user_password(
    UserPoolId=pool_id,
    Username=username,
    Password=permanent_password,
    Permanent=True
)

# Step 5: Authenticate user and get Bearer Token
print("Authenticating user and retrieving access token...")
auth_response = cognito.initiate_auth(
    ClientId=client_id,
    AuthFlow='USER_PASSWORD_AUTH',
    AuthParameters={
        'USERNAME': username,
        'PASSWORD': permanent_password
    }
)

access_token = auth_response['AuthenticationResult']['AccessToken']

# Output results
print("\n✅ Setup Complete")
print(f"Pool ID: {pool_id}")
print(f"Discovery URL: https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/openid-configuration")
print(f"Client ID: {client_id}")
print(f"Bearer Token: {access_token}")
