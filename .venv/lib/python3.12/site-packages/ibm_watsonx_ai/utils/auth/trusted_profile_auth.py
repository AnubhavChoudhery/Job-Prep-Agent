#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ibm_watsonx_ai.utils.auth import IAMTokenAuth
from ibm_watsonx_ai.utils.utils import _requests_retry_session
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.utils.auth.base_auth import (
    TokenInfo,
    STATUS_FORCELIST,
    RefreshableTokenAuth,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class TrustedProfileAuth(RefreshableTokenAuth):
    """Trusted profile authentication method class.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing, optional

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing, optional
    """

    def __init__(
        self,
        api_client: APIClient,
        on_token_creation: Callable[[], None] | None = None,
        on_token_refresh: Callable[[], None] | None = None,
    ) -> None:
        RefreshableTokenAuth.__init__(
            self, api_client, on_token_creation, on_token_refresh
        )
        self._trusted_profile_id = api_client.credentials.trusted_profile_id

        def _on_token_refresh():
            self._save_token_data(self._generate_token())

        self._internal_auth_method = IAMTokenAuth(
            api_client,
            on_token_refresh=_on_token_refresh,
        )

    def get_token(self) -> str:
        """Returns the token. If the token will be about to expire, it will be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        self._internal_auth_method.get_token()  # trigger internal token refresh if needed, before refreshing profile token if needed
        return super().get_token()

    def _generate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        response = _requests_retry_session(status_forcelist=STATUS_FORCELIST).post(
            self._href_definitions.get_iam_token_url(),
            params={
                "grant_type": "urn:ibm:params:oauth:grant-type:assume",
                "access_token": self._internal_auth_method.get_token(),
                "profile_id": self._trusted_profile_id,
            },
            headers=headers,
        )

        if response.status_code == 200:
            return TokenInfo(response.json().get("access_token"))
        else:
            raise WMLClientError(
                "Error getting trusted profile IAM Token.", response.text
            )
